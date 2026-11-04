"""
terminal.py: what you see in the terminal while agents run
"""
import threading
import time

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

COLORS = ["cyan", "magenta", "green", "yellow", "blue", "red"]

def hard_breaks(text: str) -> str:
    """markdown joins lines that a chat reply means to keep apart (outside code blocks)"""
    out, fence = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            fence = not fence
        out.append(line if fence or not line.strip() else line + "  ")
    return "\n".join(out)

class TerminalView:
    """
    prints agent events (CodeAgent's on_event) as readable, nested terminal output.
    markdown=True renders the top agent's final answer as markdown (a chat reply)
    show_task=False leaves out the top agent's task panel (in a chat you just typed it)
    """

    def __init__(self, console: Console | None = None, show_stream: bool = True, markdown: bool = False,
                 show_task: bool = True):
        self.console = console or Console(highlight=False)
        self.show_stream, self.markdown, self.show_task = show_stream, markdown, show_task
        self.live: Live | None = None
        self.stream = ""
        self.lock = threading.Lock()

    def _print(self, depth: int, *renderables):
        for r in renderables:
            self.console.print(Padding(r, (0, 0, 0, 4 * depth)))

    def _stop_stream(self):
        if self.live is not None:
            self.live.stop()
            self.live, self.stream = None, ""

    def __call__(self, m: dict):
        with self.lock:
            self._handle(m)

    def _handle(self, m: dict):
        kind, depth, agent = m["type"], m.get("depth", 0), m.get("agent", "agent")
        color = COLORS[depth % len(COLORS)]
        if kind == "run_start":
            if depth == 0 and not self.show_task:
                return
            self._print(depth, Panel(Text(m["task"][:1500]), title=f"[bold]{agent}[/] · task",
                                     title_align="left", border_style=color))
        elif kind == "llm_start":
            if m.get("step") is not None:
                self._print(depth, Rule(f"[bold]{agent}[/] · step {m['step']}", style=color, align="left"))
            if self.show_stream and self.console.is_terminal:
                label = "planning" if m.get("step") is None else "thinking"
                self.live = Live(Text(f"{label}...", style="dim"), console=self.console, transient=True,
                                 refresh_per_second=8)
                self.live.start()
        elif kind == "delta":
            if self.live is not None:
                self.stream += m["text"]
                tail = "\n".join(self.stream.splitlines()[-12:])
                self.live.update(Padding(Text(tail, style="dim"), (0, 0, 0, 4 * depth)))
        elif kind == "llm_end":
            self._stop_stream()
        elif kind == "plan":
            self._print(depth, Panel(Markdown(m["plan"]), title=f"[bold]{agent}[/] · plan", title_align="left",
                                     border_style=color, box=box.ROUNDED))
        elif kind == "action":
            parts = []
            if m.get("thought"):
                parts.append(Text(m["thought"].strip(), style="italic"))
            if m.get("code"):
                parts.append(Panel(Syntax(m["code"], "python", theme="monokai", word_wrap=True),
                                   title="code", title_align="left", border_style="dim", box=box.HORIZONTALS))
            self._print(depth, *parts)
        elif kind == "tool":
            self._print(depth, Text(f"→ {m['name']}({m.get('args', '')})", style=f"bold {color}"))
        elif kind == "step":
            if m["final"] and not m["observation"]:  # the final answer panel follows, nothing to show
                return
            obs = m["observation"] or "(no output)"
            style, title = ("red", "error") if m["error"] else ("dim", "observation")
            self._print(depth, Panel(Text(obs[-3000:]), title=title, title_align="left", border_style=style,
                                     box=box.ROUNDED))
        elif kind == "final":
            stats = (f"{m.get('state', '')} · {m.get('steps', '?')} steps · {m.get('seconds', 0):.0f}s · "
                     f"{m.get('prompt', 0):,} prompt tokens ({m.get('cached', 0):,} cached)")
            answer = Markdown(hard_breaks(str(m["answer"]))) if self.markdown and depth == 0 else Text(str(m["answer"]))
            self._print(depth, Panel(answer, title=f"[bold]{agent}[/] · final answer",
                                     title_align="left", subtitle=stats, subtitle_align="left",
                                     border_style="green" if m.get("state") == "success" else "yellow"))

class ConfirmCode:
    """
    CodeAgent's confirm hook for the terminal: the code was just printed (TerminalView),
    so ask whether to run it. y runs it, n skips it (and asks for a note the agent will
    read: why, or what to do instead)
    """

    def __init__(self, console: Console | None = None):
        self.console = console or Console(highlight=False)
        self.always = False

    def __call__(self, code: str, agent) -> str | None:
        if self.always:
            return None
        while True:
            answer = self.console.input(Text.assemble(("run this code? ", "bold yellow"),
                                                      ("[Y]es / [n]o / [a]lways ", "dim"))).strip().lower()
            if answer in ("", "y", "yes"):
                return None
            if answer in ("a", "always"):
                self.always = True
                return None
            if answer in ("n", "no"):
                return self.console.input(Text.assemble(("note for the agent ", "bold yellow"),
                                                        ("(optional) ", "dim"))).strip()
            
class SwarmTable:
    """
    The live terminal view of a swarm
    """

    def __init__(self, title: str, total: int, console: Console | None = None):
        self.console = console or Console(highlight=False)
        self.title, self.total = title, total
        self.rows: dict = {}
        self.done = self.correct = 0
        self.t0 = time.time()
        self.lock = threading.Lock()
        self.live = Live(self._render(), console=self.console, refresh_per_second=4, transient=True,
                         get_renderable=self._render)

    def __enter__(self):
        self.live.start()
        return self

    def __exit__(self, *exc):
        self.live.stop()

    def event(self, slot, m: dict):
        """an agent event from the agent(s) in this slot"""
        with self.lock:
            r = self.rows.setdefault(slot, dict(task="", stack=[], step="", status="starting", t0=time.time()))
            kind, depth = m["type"], m.get("depth", 0)
            if kind == "task":
                r.update(task=m.get("name", ""), title=m.get("title", ""), stack=[], step="", status="starting",
                         t0=time.time())
                return
            if kind == "run_start":  # who is talking: the manager, or the team member it called
                r["stack"] = r["stack"][:depth] + [m.get("agent", "agent")]
            if kind == "llm_start":
                r["status"] = "planning" if m.get("step") is None else "thinking"
                if m.get("step") is not None:
                    r["step"] = m["step"]
            elif kind == "tool":
                r["status"] = f"→ {m['name']}"
            elif kind == "action":
                r["status"] = "running code" if m.get("code") else "no code"
            elif kind == "step":
                r["status"] = "error, retrying" if m["error"] else ("submitted" if m["final"] else "ran code")
            elif kind == "final":
                r["status"] = f"done ({m.get('state', '')})" if depth == 0 else f"{m.get('agent')} answered"
                r["stack"] = r["stack"][:max(depth, 1)]
            elif kind == "agent_done":
                r["status"] = m["summary"]

    def finished(self, line: str, correct: bool):
        """a task is done: print its line above the table and count it"""
        with self.lock:
            self.done += 1
            self.correct += bool(correct)
        self.live.console.print(line, highlight=False)

    def _render(self):
        t = Table(box=box.SIMPLE_HEAD, expand=False, show_edge=False, pad_edge=False)
        for col in ("slot", "task", "agent", "step", "status", "time"):
            t.add_column(col, no_wrap=True, overflow="ellipsis", max_width=40 if col == "task" else 28)
        now = time.time()
        for slot, r in sorted(self.rows.items(), key=lambda kv: (isinstance(kv[0], str), kv[0])):
            t.add_row(str(slot), r["task"], " › ".join(r["stack"]), str(r["step"]), r["status"], f"{now - r['t0']:.0f}s")
        score = f"{self.correct}/{self.done} correct" if self.done else "no results yet"
        footer = Text(f"{self.title} · {self.done}/{self.total} done · {score} · {now - self.t0:.0f}s", style="bold")
        return Group(t, footer)
