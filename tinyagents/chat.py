"""
Terminal chatter (assumes endpoint already running on localhost:8000)

    python tinyagents/chat.py                          # a Python agent: sealed session, no files, no internet
    python tinyagents/chat.py --web                    # + web_search and visit_webpage
    python tinyagents/chat.py --web --researcher       # the web through a team member it calls (multi-agent)
    python tinyagents/chat.py --workspace ~/scratch    # + list / read / write / edit files in that folder,
                                                       #   and run Python scripts from it
    python tinyagents/chat.py --confirm                # you approve each code block before it runs
    python tinyagents/chat.py --web -t "who won the 2026 Tour de France?"   # one task, then exit
"""
import argparse
import os
import signal
import sys
try:
    import readline
except ImportError:
    pass

from rich.console import Console
from rich.syntax import Syntax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tinyagents as ta
from terminal import ConfirmCode, TerminalView
from tools import visit_webpage, web_search, workspace_tools

INSTRUCTIONS = """You are assisting a user in an interactive terminal session.

## Conversation
- Each task is a message from the user. Messages may introduce new tasks or follow up on previous ones.
- Treat the conversation as continuous. Use relevant context, previous decisions, and results from earlier messages.
- Your Python session is persistent across messages. Variables, functions, classes, and imports from previous tasks remain available.
- Do not repeat work that has already been completed unless necessary.

## Responding
- Your final answer is written for the user, not merely a raw Python result. Use clear, concise language and Markdown when appropriate.
- Answer the actual request and respect any requested output format.
- If a request is ambiguous and the ambiguity materially affects the answer, ask a clarifying question rather than guessing.
- If the task is sufficiently clear, make reasonable assumptions and proceed without unnecessary questions.
- Be honest about limitations, uncertainty, and incomplete results."""

WEB_NOTE = """
## Web research

When a task requires external, current, or verifiable information, use {how} to look it up
rather than relying on memory.

- Search for authoritative, relevant, and up-to-date sources.
- Verify important claims against the retrieved sources.
- Prefer primary sources when available, such as official documentation, papers, and datasets.
- Do not invent facts, URLs, or citations.
- Cite sources in your final answer wherever you use information obtained from them.
- Make it clear when information is uncertain, unavailable, or could not be verified.
- If the task does not require external information, do not browse unnecessarily."""

WORKSPACE_NOTE = """
## Workspace

You have access to a workspace folder through tools such as `list_files`, `read_file`,
`edit_file`, and other file operations.

- Before making changes, inspect the workspace to understand its structure and existing contents.
- Read relevant files before modifying them. Do not assume how existing code works.
- Follow the project's existing conventions, architecture, and style.
- Prefer small, targeted edits with `edit_file` when modifying existing files.
- Avoid rewriting entire files when a localized change is sufficient.
- Do not overwrite or delete existing work unless explicitly required by the task.
- Create new files only when necessary.
- After making changes, inspect the affected files and verify the result where practical.
- If an operation fails, inspect the error and adjust your approach rather than blindly retrying.
- Do not claim changes were made unless the corresponding operations succeeded.

In your final answer, summarize what you changed and explicitly list the files you created,
modified, or deleted. If no files were changed, say so."""

HELP = """[bold]commands[/]
  /new          start over: new conversation, new Python session
  /confirm      approve each code block before it runs (toggles, now {confirm})
  /steps N      the most steps the agent takes per message (now {steps})
  /save FILE    save the conversation (the agent's trajectory) as JSON
  /code         every code block the agent ran, as one script
  /tools        what the agent can use
  /exit         quit (or Ctrl-D)
A message can span lines: end a line with \\, or put \"\"\" on a line before and after it.
While the agent works, Ctrl-C stops it after the current model call; twice aborts."""

class Chat:
    def __init__(self, a, console: Console):
        self.a = a # args
        self.console = console 
        self.view = TerminalView(console, markdown=True, show_task=False)
        self.confirmer = ConfirmCode(console)
        self.confirming = a.confirm
        self.steps = a.max_steps
        extra_body = None if a.thinking else {"chat_template_kwargs": {"enable_thinking": False}}
        self.model = ta.OpenAIModel(
            a.model or "", 
            base_url=a.base_url, 
            max_tokens=a.max_tokens,
            temperature=a.temperature, 
            extra_body=extra_body
        )

        # if a model is not specified, then pick the server thats running
        if not a.model:
            self.model.model = self.model.client.models.list().data[0].id

        # get the agent up and running
        self.agent = self.build()
        self.fresh = True # the next message starts a new conversation

    def confirm(self, code, agent):
        return self.confirmer(code, agent) if self.confirming else None

    def build(self):

        a = self.a

        # default imports are all we need for things like benchmarks (humaneval, gsm8k) to do
        # things like math, algos, data structures, etc... On the other hand chat needs a little
        # more functionality. This will be something to expand on in the future for more functionality
        imports = ta.DEFAULT_IMPORTS + ["json", "csv", "textwrap", "difflib", "calendar"]
        imports += [m for m in (a.imports or "").split(",") if m]

        common = dict(
            on_event=self.view, 
            verbose=False, 
            confirm=self.confirm, 
            text_answers=True,
            step_timeout=a.step_timeout,
            venv=a.venv, 
            memory_mb=a.memory_mb
        )

        tools, agents = [], []
        instructions = INSTRUCTIONS

        if a.web:
            # if we are in research mode then we can trigger a separate web_researcher agent to search, otherwise
            # we can just use web_search or visit_webpage methods
            instructions += WEB_NOTE.format(how="web_researcher" if a.researcher else "web_search and visit_webpage")
            if a.researcher:
                agents.append(ta.CodeAgent(
                    self.model, 
                    tools=[web_search, visit_webpage], 
                    max_steps=10, 
                    name="web_researcher",
                    description="Searches the web and reads pages to answer a question; returns the facts "
                                "it found with their source URLs.", 
                                **common))

            else:
                tools += [web_search, visit_webpage]

        if a.workspace:
            ws = workspace_tools(a.workspace, gpu=a.gpu, venv=a.venv, timeout=a.run_timeout)

            # only keep file reading
            if a.read_only:
                ws = [t for t in ws if t.name in ("list_files", "read_file")]

            tools += ws
            instructions += WORKSPACE_NOTE

        return ta.CodeAgent(
            self.model, 
            tools=tools, 
            agents=agents, 
            imports=imports, 
            max_steps=self.steps,
            instructions=instructions, 
            planning_interval=a.planning_interval, 
            **common
        )

    def banner(self):
        """just a printout"""
        a, agent = self.a, self.agent
        c = self.console
        c.print(f"[bold]tinyagents chat[/] · {self.model.model} at {a.base_url}")
        c.print(f"  python session: sandboxed (no files, no network){' + ' + a.venv if a.venv else ''}"
                f" · imports {', '.join(agent.imports)}", style="dim")
        if a.workspace:
            c.print(f"  workspace: {os.path.realpath(a.workspace)} ({'read-only' if a.read_only else 'read/write'})",
                    style="dim")
        c.print(f"  tools: {', '.join(agent.tools) or 'none'}", style="dim")
        c.print("  /help for commands · Ctrl-D to quit", style="dim")

    def ask(self, text: str):
        """how we run based on the case"""
        presses = 0

        def on_sigint(*_):
            """two presses to kill"""
            nonlocal presses
            presses += 1
            if presses > 1:
                raise KeyboardInterrupt
            self.agent.interrupt()
            self.console.print("[yellow]stopping after the current model call (Ctrl-C again to abort now)[/]")

        old = signal.signal(signal.SIGINT, on_sigint)

        # A fresh run at the start
        try:
            self.agent.run(text, reset=self.fresh, max_steps=self.steps)
            self.fresh = False
        # if interrupted, picks up from here
        except ta.AgentInterrupted:
            self.fresh = False
            self.console.print("[yellow]stopped. Your next message continues from here.[/]")

        # if keyboard interrupt the sandboxes message may be broken (abort while sandbox.run() was doing stuff)
        # so we close the live streaming display, kill the worker, and start from scratch
        except KeyboardInterrupt:  
            self.view._stop_stream()
            self.agent.close()
            self.fresh = True
            self.console.print("[red]aborted. The Python session was restarted; the next message starts fresh.[/]")
       
        except Exception as e: 
            self.view._stop_stream()
            self.console.print(f"[red]{type(e).__name__}: {e}[/]")

        finally:
            signal.signal(signal.SIGINT, old)

    def command(self, line: str) -> bool:
        """a /command; False to quit"""
        cmd, _, arg = line.partition(" ")
        arg, c = arg.strip(), self.console
        if cmd in ("/exit", "/quit"):
            return False
        if cmd == "/new":
            self.agent.close()
            self.fresh = True
            c.print("[dim]new conversation[/]")
        elif cmd == "/confirm":
            self.confirming, self.confirmer.always = not self.confirming, False
            c.print(f"[dim]confirm before running code: {'on' if self.confirming else 'off'}[/]")
        elif cmd == "/steps" and arg.isdigit():
            self.steps = int(arg)
            c.print(f"[dim]up to {self.steps} steps per message[/]")
        elif cmd == "/save" and arg:
            self.agent.save(os.path.expanduser(arg))
            c.print(f"[dim]saved {len(self.agent.messages)} messages to {arg}[/]")
        elif cmd == "/code":
            code = self.agent.return_full_code()
            c.print(Syntax(code, "python", theme="monokai") if code else "[dim](no code yet)[/]")
        elif cmd == "/tools":
            for t in self.agent.tools.values():
                c.print(f"[bold]{t.name}[/]{t.signature}  [dim]{t.description.strip().splitlines()[0]}[/]")
            if not self.agent.tools:
                c.print("[dim]no tools: only its Python session[/]")
        else:
            c.print(HELP.format(confirm="on" if self.confirming else "off", steps=self.steps))
        return True

def read_message(prompt: str) -> str | None:
    """we could type a message over several lines but we need to flatten it as that is how 
    input() will read it. """
    try:
        ### a codeblock can be between """ ... """
        line = input(prompt)
        if line.strip() == '"""':
            lines = []
            while (line := input("... ")).strip() != '"""':
                lines.append(line)
            return "\n".join(lines)

        ### when there is a \ at the end of a line for continuing a sentence
        lines = []
        while line.endswith("\\"):
            lines.append(line[:-1])
            line = input("... ")
        return "\n".join(lines + [line]).strip()
    
    except EOFError:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--task", help="run this one task and exit")
    ap.add_argument("--model", default=None, help="default: the model the server is serving")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--max-tokens", type=int, default=4096, help="per model call")
    ap.add_argument("--temperature", type=float, default=None, help="default: the server's default")
    ap.add_argument("--thinking", action="store_true", help="leave Qwen3 thinking on")
    ap.add_argument("--max-steps", type=int, default=20, help="per message")
    ap.add_argument("--planning-interval", type=int, default=None, help="write a plan, update it every N steps")
    ap.add_argument("--web", action="store_true", help="give it web_search and visit_webpage")
    ap.add_argument("--researcher", action="store_true", help="with --web: a web_researcher agent it calls, "
                                                              "instead of the web tools directly")
    ap.add_argument("--workspace", help="a folder it can read and write (created if missing)")
    ap.add_argument("--read-only", action="store_true", help="with --workspace: list and read files only")
    ap.add_argument("--gpu", type=int, default=None, help="with --workspace: one GPU for scripts it runs")
    ap.add_argument("--run-timeout", type=int, default=600, help="seconds a workspace script may run")
    ap.add_argument("--venv", help="a sandbox venv (setup_sandbox_venv.sh) for its session and scripts")
    ap.add_argument("--imports", help="extra modules it may import, comma separated (e.g. numpy,pandas)")
    ap.add_argument("--step-timeout", type=int, default=60, help="seconds one code block may run")
    ap.add_argument("--memory-mb", type=int, default=4096, help="address space cap of its session")
    ap.add_argument("--confirm", action="store_true", help="approve each code block before it runs")
    a = ap.parse_args()
    if a.researcher and not a.web:
        ap.error("--researcher needs --web")

    console = Console(highlight=False)
    try:
        chat = Chat(a, console)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]can't reach a model at {a.base_url}: {type(e).__name__}: {e}[/]")
        return 1
    if a.task:
        chat.ask(a.task)
        return 0 if chat.agent.result and chat.agent.result.state == "success" else 1

    chat.banner()

    # just some text formatting
    prompt = "\001\033[1;32m\002you ›\001\033[0m\002 " if sys.stdin.isatty() else "you › "
    while True:
        try:
            text = read_message(prompt)
        except KeyboardInterrupt:  # at the prompt: drop the line, like a shell
            console.print()
            continue
        if text is None:
            break
        if not text:
            continue
        if text.startswith("/"):
            if not chat.command(text):
                break
            continue
        chat.ask(text)
    chat.agent.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
