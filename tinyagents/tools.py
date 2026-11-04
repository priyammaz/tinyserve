"""
Prebuilt tools that come with tinyagents
"""

import os
import re
import subprocess
import threading
import time

from tinyagents import Tool, _bwrap_args, _sandbox_python, tool, truncate

USER_AGENT = "Mozilla/5.0"

class _Throttle:
    """
    Intervals between requests across all threads
    """
    def __init__(self, interval):
        self.interval = interval
        self.lock = threading.Lock()
        self.last = 0.0

    def wait(self):
        with self.lock:
            delay = self.last + self.interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.last = time.monotonic()

_search_throttle = _Throttle(1.0)

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Searches the web (DuckDuckGo) and returns the top results: title, URL, and a snippet each.
    use visit_webpage(url) to read a result in full. 

    Args:
        query: what to search for, like you'd type into a search engine
        max_results: how many results to return (at most 10)
    """

    from ddgs import DDGS
    for attempt in range(3):
        _search_throttle.wait()
        try:
            results = DDGS().text(query, max_results=min(max_results, 10))
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    if not results:
        return f"No results for {query!r}. Try a shorter or different query."
    return "\n\n".join(f"[{r.get('title', '')}]({r.get('href', '')})\n{r.get('body', '')}" for r in results)

@tool
def visit_webpage(url: str, max_chars: int = 10000) -> str:
    """Downloads a web page and returns its text (as markdown when possible), cut to max_chars.

    Args:
        url: the full URL, starting with http:// or https://
        max_chars: the most characters to return (the page's start and end are kept)
    """
    import requests
    if not re.match(r"https?://", url):
        raise ValueError("url must start with http:// or https://")

    # use request to get the page
    r = requests.get(url, timeout=20, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()

    # if it isnt an html (it could be a github code file, csv, mkd, markdown, etc...)
    if "html" not in r.headers.get("content-type", "html"):
        # truncate to our max wanted length
        return truncate(r.text, max_chars)

    # Parse with beautiful soup otherwise
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(r.text, "html.parser")
    # menus, headers and footers are most of a page's text and none of its content
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg"]):
        tag.decompose()
    page = soup.find("main") or soup.find("article") or soup.body or soup

    # use to convert html content into a nice markdown file
    try:
        from markdownify import markdownify  # keeps links and headings, if installed
        text = markdownify(str(page))
    except ImportError:
        # plain text: a newline after every block element, spaces between inline ones
        for block in page.find_all(["p", "div", "li", "tr", "br", "h1", "h2", "h3", "h4", "h5", "h6",
                                    "table", "pre", "blockquote", "dt", "dd"]):
            block.insert_after("\n")
        text = page.get_text(" ")

    # cleanup
    text = re.sub(r"[ \t ﻿]+", " ", text) # collapse spaces on a line
    text = re.sub(r" *\n *", "\n", text) # trim spaces around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text).strip() # limit blank lines, trim ends

    # tuncate always
    return truncate(text, max_chars)

def workspace_tools(
    root: str, 
    gpu: int | None = None, 
    venv: str | None = None, 
    timeout: int = 600, 
    max_file_mb: int = 1024, 
    protected: tuple[str, ...] = ()
) -> list[Tool]:

    """
    These are a collection of tools confined to one folder, so paths are 
    all relative to it.  Anything that resolves outside it (..., absolute paths, symlinks) 
    is refused

        gpu:       run_python gets this one GPU (its device node only), None for none
        venv:      the Python environment run_python's sandbox uses (torch etc.)
        timeout:   seconds a run_python call may take before it's killed
        protected: files write_file refuses to touch (relative paths)

    """

    root = os.path.realpath(root)
    os.makedirs(root, exist_ok=True)

    def resolve(path: str) -> str:
        """
        throw error if a path is outside of our root dir
        """
        full = os.path.realpath(os.path.join(root, path))
        if full != root and not full.startswith(root + os.sep):
            raise PermissionError(f"{path!r} is outside the workspace")
        return full

    def limit_file_size():
        import resource
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_mb << 20, max_file_mb << 20))

    @tool
    def list_files() -> str:
        """Lists every file in your workspace with its size.
        """
        out = []
        for d, dirs, files in os.walk(root):
            dirs[:] = sorted(x for x in dirs if not x.startswith("."))
            for f in sorted(files):
                p = os.path.join(d, f)
                out.append(f"{os.path.relpath(p, root)}  ({os.path.getsize(p):,} bytes)")
        return "\n".join(out) or "(empty)"

    @tool
    def read_file(path: str, max_chars: int = 20000) -> str:
        """Reads a text file from your workspace.

        Args:
            path: the file's path, relative to the workspace
            max_chars: the most characters to return (the start and end are kept)
        """
        with open(resolve(path), encoding="utf-8", errors="replace") as f:
            return truncate(f.read(), max_chars)

    @tool
    def write_file(path: str, content: str) -> str:
        """Writes (or overwrites) a text file in your workspace, creating folders as needed.

        Args:
            path: the file's path, relative to the workspace
            content: the complete new content of the file
        """
        full = resolve(path)
        if os.path.relpath(full, root) in protected:
            raise PermissionError(f"{path} is read-only")
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"wrote {len(content):,} characters to {path}"

    @tool
    def edit_file(path: str, old: str, new: str) -> str:
        """Edits a text file in your workspace by replacing one exact snippet of it. The snippet must
        appear exactly once (include a few surrounding lines to make it unique); read the file first.

        Args:
            path: the file's path, relative to the workspace
            old: the exact text to replace, including its indentation
            new: the text to put in its place
        """
        full = resolve(path)
        if os.path.relpath(full, root) in protected:
            raise PermissionError(f"{path} is read-only")
        with open(full, encoding="utf-8") as f:
            text = f.read()
        count = text.count(old)
        if count != 1:
            raise ValueError(f"the snippet appears {count} times in {path}, it must appear exactly once"
                             + (" (check whitespace and indentation)" if count == 0 else " (include more lines)"))
        with open(full, "w", encoding="utf-8") as f:
            f.write(text.replace(old, new))
        return f"edited {path}: replaced {len(old.splitlines()) or 1} lines with {len(new.splitlines()) or 1}"

    @tool
    def run_python(path: str, args: str = "") -> str:
        """Runs a Python script from your workspace (e.g. a training run) and returns its exit code, how
        long it took, and the end of what it printed. It runs in a fresh sandbox where the workspace is
        the current directory and writable{gpu_note}; there is no internet.

        Args:
            path: the script's path, relative to the workspace
            args: command-line arguments for the script, separated by spaces
        """
        rel = os.path.relpath(resolve(path), root)
        cmd = _bwrap_args(venv=venv, writable=[(root, "/workspace")], gpu=gpu, chdir="/workspace")
        cmd += [_sandbox_python(venv), "-u", rel, *args.split()]
        t = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, preexec_fn=limit_file_size)
            code, out = r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired as e:
            code = "killed (timeout)"
            out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            out += f"\n... killed after {timeout}s"
        return f"exit code {code} after {time.time() - t:.0f}s\n{truncate(out, 6000)}"

    # overwrite the {gpu_note} in our run_python
    run_python.description = run_python.description.replace(
        "{gpu_note}", f", with one GPU (cuda:0)" if gpu is not None else "")

    # return a list of the tools we want
    return [list_files, read_file, write_file, edit_file, run_python]