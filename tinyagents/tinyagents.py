import ast
import builtins
import inspect
import json
import os
import re
import resource
import select
import shutil
import signal
import subprocess
import sys
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from types import ModuleType
from typing import Callable

DEFAULT_IMPORTS = ["math", "re", "collections", "itertools", "functools", "heapq", "string", "bisect",
                   "statistics", "fractions", "decimal", "typing", "random", "datetime", "copy", "hashlib"]
SANDBOX_PYTHON = "/usr/bin/python3"
WORKER_PATH = "/opt/tinyagents/tinyagents.py"  
VENV_PATH = "/opt/venv"                         
FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "input", "globals", "locals", "vars", "dir", "getattr", "setattr",
    "delattr", "breakpoint", "memoryview", "help", "exit", "quit", "__import__", "__builtins__",
}
FORBIDDEN_ATTRS = {
    "format", "format_map", "vformat", "get_field", "attrgetter", "methodcaller", "mro",
    "gi_frame", "gi_code", "cr_frame", "cr_code", "ag_frame", "ag_code",
    "f_back", "f_globals", "f_locals", "f_builtins", "f_code", "tb_frame", "tb_next",
}
SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "ascii", "bin", "bool", "bytes", "callable", "chr", "complex", "dict", "divmod",
    "enumerate", "filter", "float", "frozenset", "hasattr", "hash", "hex", "id", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min", "next", "object", "oct", "ord", "pow",
    "range", "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
    "property", "staticmethod", "classmethod", "super", "format", "bytearray",
    "Exception", "ArithmeticError", "AssertionError", "AttributeError", "IndexError", "KeyError",
    "LookupError", "NotImplementedError", "OverflowError", "RecursionError", "RuntimeError",
    "StopIteration", "TypeError", "ValueError", "ZeroDivisionError", "NotImplemented", "Ellipsis",
    "__build_class__", 
]

SYSTEM_PROMPT = """You are an expert Python programmer. You solve tasks by writing and executing Python code.

## Response format

Unless a note below says otherwise, every response must follow this exact format:

Thought: <brief reasoning about the next step>
<code>
<Python code>
</code>

Do not include anything outside this format.

## Execution environment

Your code runs in a persistent Python session. Variables, functions, classes, and imports persist between steps.

After each execution, you receive an Observation containing:
- Anything printed to stdout.
- The value of the final bare expression, if any.
- Any errors raised during execution.

Use observations to inspect results, debug errors, and decide what to do next.

When you have solved the task, call `final_answer(result)` in your code. This submits the result and ends the task.

## Rules

- Only these modules may be imported: {imports}.
- There is no file, network, or operating-system access{tools_note}.
- Do not access attributes beginning with an underscore.
- Do not use `eval`, `exec`, `open`, `getattr`, or `str.format`. Use f-strings instead.
- Write and execute Python to perform computations rather than attempting to calculate complex results mentally.
- Keep each execution step focused and reasonably small.
- Print intermediate results only when they help you inspect or verify something.
- If execution produces an error, inspect it and correct your code. Do not blindly repeat failed attempts.
- Verify the result when practical before submitting it.
- Do not call `final_answer` until you have a complete answer.

## Example

Task: What is the 10th Fibonacci number?

Thought: I'll implement an iterative Fibonacci function and check the requested value.

<code>
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print(fib(10))
</code>

Observation: 55

Thought: The computation returned 55, so I can submit the result.

<code>
final_answer(55)
</code>
{tools}{instructions}"""

TOOLS_SECTION = """

## Available tools

The following functions are available in your Python session. Call them like ordinary Python
functions, they are already defined, so do not redefine them.

Tool calls execute outside the Python sandbox and return their results to your code. Print
the results when you need to inspect them.

If a tool call fails, it raises `ToolError`. You may catch this exception and handle the
failure, or let it propagate and inspect the resulting Observation.

Available tool signatures:
<code>
{stubs}
</code>"""

TEXT_ANSWER_NOTE = """

## Direct answers

Not every task requires code. If you can answer a question directly from your existing
knowledge, or the user is simply having a conversation, respond in plain text without
a Thought or code block. This is your final answer.

If the task requires computation, tool use, or external information, use the normal
Thought/code execution loop instead. Do not provide a premature answer while further
work is needed."""

TEAM_NOTE = """

## Team members

Some available functions represent other agents. Their descriptions identify them as team
members.

You can delegate tasks to these agents by calling their functions. Each agent operates
independently and cannot see your conversation or previous observations.

Give each agent a complete, self-contained task with all the context and information it needs.
Their function returns their final answer as a string, which you can inspect and use in your
own work."""

PLAN_PROMPT = """Before writing any code, analyze the task and make a plan.

Task:
{task}

Identify what is already known, what information is missing, and how you will obtain or derive it.

Respond in exactly this format:

## Facts

### Given
Facts and constraints explicitly provided in the task.

### Known
Relevant information you already know and can rely on.

### To look up
Information that requires external research or tool calls. If nothing is needed, write "None."

### To derive
Values or conclusions that must be obtained through computation or reasoning. If nothing is needed, write "None."

## Plan
A short, numbered sequence of high-level steps describing how you will solve the task. Use the available tools where appropriate. Do not include code.

Finish with <end_plan> on its own line and stop. Do not execute code or attempt to answer the task yet."""

UPDATE_PLAN_PROMPT = """Pause execution and update your plan. You have {remaining} steps remaining.

Review the task, your original plan, all code executions, observations, tool results, and conclusions so far.

Reassess the task:
- Update the facts based on everything learned so far.
- Remove questions that have already been answered.
- Identify remaining unknowns, unresolved computations, and outstanding requirements.
- Account for any errors, failed tool calls, or changes in approach.
- Prioritize the remaining work given the available steps.

Respond in exactly the same format as before:

## Facts

### Given
Facts and constraints explicitly provided in the task.

### Known
Relevant information established so far, including results from computations and tool calls.

### To look up
Information that still requires external research or tool calls. If nothing remains, write "None."

### To derive
Values or conclusions that still require computation or reasoning. If nothing remains, write "None."

## Plan
A short, numbered sequence of high-level steps for completing the remaining work. Do not repeat completed steps.

Do not include code or attempt to execute anything.

Finish with <end_plan> on its own line and stop."""

RETRY_HINT = (
    "\nThe previous attempt failed. Analyze the error and the approach that caused it before "
    "trying again. Fix the underlying problem rather than repeating the same code. "
    "If the same approach has failed multiple times, switch to a fundamentally different "
    "approach. Avoid repeating work that has already succeeded."
)

MAX_STEPS_PROMPT = """You have exhausted your execution steps and cannot run any more code or call any more tools.

Based on the task, all previous reasoning, code executions, observations, and tool results, provide your best final answer.

Task:
{task}

Requirements:
- Answer the original task directly.
- Use the results and evidence already available.
- Do not claim to have performed computations or lookups that were not completed.
- If information is incomplete, be transparent about uncertainty rather than inventing results.
- Follow the task's requested format exactly. If it asks for source code, provide only the complete code.
- Do not include your reasoning, execution history, or commentary.

Return only the final answer."""

STOP = ["</code>", "Observation:"]


class CodeCheck(ast.NodeVisitor):
    """
    Static check of a code block before it runs, returns any readable errors
    so the agent can fix its code. This is a wrapper on 

        ast.NodeVisitor
    
    which allows us to traverse syntax trees and check for things we want. Any method
    we want to run as a check must start with visit_* and ends with the attribute we want
    to take a look at (Name, Attribute, FunctionDef, Import, etc...)
    """

    def __init__(self, imports, reserved=frozenset()):
        self.imports = imports
        self.errors = []
        self.reserved = reserved

    def fail(self, node, msg):
        """
        quick helper to get the line number + error of a message. If line number
        is not available we just use a ?
        """
        err_msg = f"line {getattr(node, 'lineno', '?')}: {msg}"
        self.errors.append(err_msg)

    def _check_rebind(self, node, name):
        """
        We need to protect reserved names. For example, some reserved things we will have are:
            reserved = {"final_answer", "search"}

        then we are allowed to:
            result = search(weather)
            final_answer(result)

        But we are not allowed to:
            search = 123
            def final_aswer(results):
                print(result)

        We will ALWAYS have final_answer as that always ends the task, and we also will have
        a collection of tools available that we dont want the code to overwrite            
        """
        if name in self.reserved:
            what = "the function that ends the task" if name == "final_answer" else "a tool"
            self.fail(node, f"'{name}' is {what}, already defined in your session: don't define or assign it, "
                            f"just call it, e.g. result = {name}(...)")
    
    def visit_Name(self, node):
        """
        Name will inspect ast.Name objects, which are the variable names. For example lets say
            x = 10
            print(x)
            del x
        
        All three of these occurances of x are represented by ast.Name nodes, but they have different
        contexts
            x = 10 -> ast.Store
            print(x) -> ast.Load
            del x -> ast.Del

        We will ensure that if the identifier is listed in FORBIDDEN NAMES or is a dunder, then its unsafe
        If it is storing / deleting, we will ensure its not overwriting/deleting our tools or reserved names
        """
        if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
            self.fail(node, f"'{node.id}' is not allowed")
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._check_rebind(node, node.id)
        self.generic_visit(node) # continue recursion, but Name doesnt typically have children

    def visit_FunctionDef(self, node):
        """
        The name of a newly defined function shouldnt overwrite one of our tools or reserved names
        """
        self._check_rebind(node, node.name)
        self.generic_visit(node)

    # async def is the same thing
    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        """
        Any new class definitions should't overwrite our reserved names
        """
        self._check_rebind(node, node.name)
        self.generic_visit(node)

    def visit_arg(self, node):
        """
        If the argument to a function is search lets say

            def calculate(search):
                return search
        
        this would overwite "search" from our reserved local to the function which
        we also dont want to allow. This wouldnt be found by Name, as the name is
        calculate, but search is the arg to it
        """
        self._check_rebind(node, node.arg)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """
        control attribute access (like pandas.DataFrame), this is just our
        attr assined to the object. We block a few cases:

            1. starts with _, these are private/dunder
            2. block any forbidden attributes
        """
        if node.attr.startswith("_"):
            self.fail(node, f"attribute '{node.attr}' is not allowed (names starting with _ are blocked)")
        elif node.attr in FORBIDDEN_ATTRS:
            hint = " (use an f-string)" if "format" in node.attr else ""
            self.fail(node, f"attribute '{node.attr}' is not allowed{hint}")
        self.generic_visit(node)

    def _check_module(self, node, name):
        """
        Check if an imported module is allowed. For example:
            torch.nn.functional
        this gets split on "." and we take the first, so basically, is torch allowed?
        
        """
        if name.split(".")[0] not in self.imports:
            self.fail(node, f"import of '{name}' is not allowed (allowed: {', '.join(sorted(self.imports))})")

    def visit_Import(self, node):
        """
        Check our imports. in import numpy as np, "name" is numpy but "asname" is np

        We reject imports that are not in our accepted list, and renames of things like
        import numpy as search, as that would conflict
        """
        for alias in node.names:
            self._check_module(node, alias.name)
            self._check_rebind(node, alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node):
        """
        Same as before but handling from ... import ...

        For example from torch.nn import Linear
            node.module: torch.nn
            node.level: 0
            node.names: [alias(name="Linear", asname=None)]

        We reject relative imports, things like from . import utils has 
        a node level != 0
        """
        if node.level:
            self.fail(node, "relative imports are not allowed")
        # this node.module or "" is because from . import ... has no module, though we catch earlier
        self._check_module(node, node.module or "") 
        for alias in node.names:
            if alias.name == "*":
                self.fail(node, "star imports are not allowed")
                continue
            if alias.name.startswith("_"): # dont allow things like from module import _private
                self.fail(node, f"importing '{alias.name}' is not allowed")
            self._check_rebind(node, alias.asname or alias.name) # check for conflict with reserved
        
def check_code(code, imports, reserved=frozenset()):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"
    
    checker = CodeCheck(set(imports), set(reserved))
    checker.visit(tree)
    return "\n".join(checker.errors) or None

class SafeModule:
    """
    The CodeCheck allows us to inspect Python source code before running it, but of course 
    we dont capture everything possible. So we also create the SafeModule that controlls
    what can happen WHILE the code is running. This is mostly about trying to make sure
    we cant access things we dont want (like private/dunder methods)
    """
    def __init__(self, module, imports):
        object.__setattr__(self, "_m", module)
        object.__setattr__(self, "_imports", imports)

    def __getattr__(self, name):
        m = object.__getattribute__(self, "_m")
        imports = object.__getattribute__(self, "_imports")

        # block private/dunder
        if name.startswith("_"):
            raise AttributeError(f"'{m.__name__}.{name}' is not allowed")

        # get the actual attribute
        value = getattr(m, name) # like if we did numpy.pi, this is pi

        # The attribute could itself be a module, so check as a safemodule
        if isinstance(value, ModuleType):
            # lets say we dont want sys in our imports. well that means our CodeCheck will
            # flag it if we do import sys. But what if we do typing.sys?
            if value.__name__.split(".")[0] not in imports:
                raise AttributeError(f"'{m.__name__}.{name}' is not allowed")
            return SafeModule(value, imports)
        return value

    def __setattr__(self, name, value):
        ### Cannot add stuff to a module
        raise AttributeError("modules are read-only here")

    def __repr__(self):
        ### normal printout
        return f"<module {object.__getattribute__(self, '_m').__name__}>"

class FinalAnswer(BaseException):
    """
    This is the signal that the agent has produced its final answer. 

    The exception is used to stop code execution and return a value to the caller
    and it inherits BaseException so taht ordinary except Exception as e lines
    dont accidentally capture and supress it
    """
    def __init__(self, value):
        self.value = value

class ToolError(Exception):
    """Exception for when a Tool failed (bad args, network error, etc...)
    and then the agent code can catch it"""

def safe_builtins(imports, output):
    """
    Replace common methods like print, import, etc... with our own version
    """
    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        # if level != 0 or import not in allowed list raise
        if level or name.split(".")[0] not in imports:
            raise ImportError(f"import of '{name}' is not allowed")

        # Actually import the module (this is what happens when you do import torch)        
        module = builtins.__import__(name, globals, locals, fromlist, level)

        # return the module as a safemodule
        return SafeModule(module, imports)

    def safe_print(*args, sep=" ", end="\n"):
        # instead of printing to terminal store in a list
        output.append(sep.join(map(str, args)) + end)

    def final_answer(value):
        # raise for final answer
        raise FinalAnswer(value)

    # replace the global vars for these methods with our own
    env = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}
    env.update(__import__=safe_import, print=safe_print, final_answer=final_answer, ToolError=ToolError)

    return env

def _limit_resources(memory_mb, cpu_seconds):
    """
    apply operating system resource limits to the current process
    """
    limits = {
        resource.RLIMIT_NPROC: 1,                        # no fork / subprocess / threads
        resource.RLIMIT_AS: memory_mb << 20,             # address space (convert mb to bytes)
        resource.RLIMIT_CPU: cpu_seconds,                # total CPU seconds for this process
        resource.RLIMIT_FSIZE: 0,                        # no writing files (nothing is writable anyway)
        resource.RLIMIT_NOFILE: 32,                      # limit open file descriptors
        resource.RLIMIT_CORE: 0,                         # no core dumps
    }
    for which, value in limits.items():
        resource.setrlimit(which, (value, value))  # hard limit too: can't be raised again

def truncate(text, limit):
    """keep the start AND the end: errors and results are usually at the end"""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... ({len(text) - limit} characters cut) ...\n{text[-half:]}"

def _agent_lineno(e: BaseException):
    """
    Walk the traceback of an exception and return the line number of the 
    deepest frame that coms from agents code. later in our _execute our errors
    will print out like: 

    traceback frames:
        frame from some library     -> skip
        frame from <agent> line 3   -> lineno = 3
        frame from <agent> line 7   -> lineno = 7   <- final value
        frame from user code        -> skip
    """
    tb, lineno = e.__traceback__, None
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "<agent>":
            lineno = tb.tb_lineno
        tb = tb.tb_next
    return lineno


def _execute(code: str, namespace: dict, output: list, cfg: dict, timeout: int):
    output.clear() # clear anything currently in here before execution

    # check the code
    error = check_code(code, cfg["imports"], {"final_answer", *cfg["tools"]}) if cfg["check"] else None

    final, value = False, None
    if error is None:

        # raise if alarm
        def on_alarm(*_):
            raise TimeoutError(f"step exceeded {timeout}s")
        signal.signal(signal.SIGALRM, on_alarm)
        signal.alarm(timeout) # start timer

        try:
            # parse the agent's python code into the same abstracy syntax tree from earlier
            # lets say the code is:
            #   x = 10
            #   x + 5
            # we have an Assign node for x = 10
            # Expr node for x + 5
            tree = ast.parse(code)

            # if the final statement is an expression, we remove it from the AST and save
            # it seperately, which allows us to eval the expression and display its result
            # print() is also an ast.Expr node
            #   last = Expr(x+5)
            #   tree.body = [Assign(x = 10)]
            last = (
                tree.body.pop()
                if tree.body and isinstance(tree.body[-1], ast.Expr)
                else None
            )

            # compile and execute the remaining AST
            # in this case "x" will be added to the namespace (namespace["x"] = 10)
            # <agent> is just the filename so if we get errors we get:
            # File "<agent>", line 7, in foo

            exec(compile(source=tree, filename="<agent>", mode="exec"), namespace)

            # If we had our final line afterwards
            if last is not None:

                # eval the final line using the same namespace
                shown = eval(compile(ast.Expression(last.value), "<agent>", "eval"), namespace)
                if shown is not None:
                    # add to the output
                    output.append(f"Out: {shown!r}\n")

        except FinalAnswer as f:
            # If at some point during the code execution there is the final_answer()
            # we will raise our special FinalAnswer which is captured here
            final, value = True, f.value

        except BaseException as e:
            # Any execution failure is returned back
            error = f"{type(e).__name__}: {e}"
            lineno = _agent_lineno(e)
            # extract the line cooresponding to the number if available
            if lineno:
                line = code.splitlines()[lineno - 1].strip() if lineno <= len(code.splitlines()) else ""
                error = f"line {lineno} `{line}`: {error}"

        finally:
            # if done turn the alarm off everything happened in timeout
            signal.alarm(0)

    # create our final output
    text = truncate("".join(output), cfg["max_output"])

    # ensure the output is json compatible. For example if the answer is 42, [1,2,3], {"answer": 42}
    # there is no issues. But if the answer is returned as an object, then we have a problem as json
    # will throw a type error
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        value = repr(value) # in which case make value the string representation of that object

    return {"output": text, "error": error, "final": final, "value": value}

def _worker_main():
    """
    read {"code": ...} lines and run them in one persistent namespace
    """
    # standard output stream for prints, etc...
    protocol_out = sys.stdout

    # our config will be passed in through stdin later via stdin.write(json.dumps(cfg))
    cfg = json.loads(sys.stdin.readline())

    # set resource limits
    _limit_resources(cfg["memory_mb"], cfg["cpu_seconds"])

    output = []
    imports = set(cfg["imports"])

    if cfg["check"]:
        env = safe_builtins(imports, output)
    else:
        # no guardrails, all ordinary python, only the OS sandbox around it
        env = dict(vars(builtins))
        # manually change stuff we need still
        env.update(
            # print should write to output
            print=lambda *a, sep=" ", end="\n": output.append(sep.join(map(str, a)) + end),
            # final answer should raise https://stackoverflow.com/questions/8294618/define-a-lambda-expression-that-raises-an-exception
            final_answer=lambda v: (_ for _ in ()).throw(FinalAnswer(v)), 
            # ToolError should be available as a builtin
            ToolError=ToolError
        )

    # the namespace for this run
    namespace = {"__builtins__": env, "__name__": "__agent__"}

    def call_host(name, args, kwargs):
        """
        Agent runs in a sandbox, but it can run tools on the host and wait for the result

        It will call the tool name with positional args and keyword kwargs

        agent  --(JSON line on stdout)-->  host
        agent  <--(JSON line on stdin)--   host

        """
        try:
            # serialize the toolcall into a single json line
            request = json.dumps({"tool": name, "args": list(args), "kwargs": kwargs})
        except (TypeError, ValueError):
            raise TypeError(f"{name}(): arguments must be plain data (str, numbers, lists, dicts)") from None

        # clear our timer, this pauses the steps timeout as some tools (like websearch) doesnt get killed mid request
        remaining = signal.alarm(0)

        try:
            # write the json to the host via print
            protocol_out.write(request+"\n")
            protocol_out.flush() # force write immediately
            reply = json.loads(sys.stdin.readline()) # waits until host writes one full line ending in \n
        finally:
            # set timer again after toolcall
            if remaining:
                signal.alarm(remaining)

        if reply.get("error") is not None:
            raise ToolError(reply["error"])

        return reply["result"]

    def make_stub(name):
        """
        A fake function that just forwards the call to the host
        """
        def stub(*args, **kwargs):
            return call_host(name, args, kwargs)

        # this is just so if somethign goes wrong and it prints, we 
        # get the name of the actual tool not the "stub" method
        stub.__name__ = stub.__qualname__ = name

        return stub

    for name in cfg["tools"]:
        namespace[name] = make_stub(name)

    # send one json line to the host saying the worker is ready
    protocol_out.write(json.dumps({"ready": True, "python": sys.version.split()[0]}) + "\n")
    protocol_out.flush()

    # the workers main request loop
    while line := sys.stdin.readline(): # waits until some line comes in
        req = json.loads(line)
        # any variables we want in the request can be added to the namespace
        if "vars" in req:  # data handed in by the host (CodeAgent.run(inputs=...))
            namespace.update(req["vars"])
            resp = {"ok": True}
        else:
            resp = _execute(req["code"], namespace, output, cfg, req["timeout"])

        # write back the response
        protocol_out.write(json.dumps(resp) + "\n")
        protocol_out.flush()

def _sandbox_python(venv):
    """
    Which python to use?
    """
    return f"{VENV_PATH}/bin/python" if venv else SANDBOX_PYTHON

def gpu_devices(index: int) -> list[str]:
    """the device nodes one NVIDIA GPU needs: its own node plus the shared control nodes"""
    nodes = [f"/dev/nvidia{index}", "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"]
    return [n for n in nodes if os.path.exists(n)]

def _bwrap_args(extra_binds: list[tuple[str, str]] = (), 
                venv: str | None = None,
                writable: list[tuple[str, str]] = (), 
                gpu: int | None = None, chdir: str = "/",
                tmp: bool = False, 
                env: dict | None = None):
    """
    extra_binds: read-only (host path, sandbox path) pairs
    venv:        a Python environment, read-only at VENV_PATH
    writable:    (host path, sandbox path) pairs the sandbox may write to
    gpu:         pass this one NVIDIA GPU through: only its device node (plus the shared control
                 nodes), so inside it's the only GPU, cuda:0. CUDA also reads the PCI topology,
                 so /sys comes along read-only
    tmp:         a private, empty, in-memory /tmp (gone when the sandbox exits)
    env:         extra environment variables

    More information about bubblewrap can be found here: https://github.com/containers/bubblewrap
    """
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError("tinyagents runs code only inside bubblewrap, which isn't installed "
                           "(sudo apt install bubblewrap). isolation='local' runs it without, if you must.")
    args = [bwrap,
            "--unshare-all",                 
            "--die-with-parent", "--new-session", "--cap-drop", "ALL",
            "--clearenv", "--setenv", "PATH", "/usr/bin", "--setenv", "HOME", "/",
            "--ro-bind", "/usr", "/usr"]
    for d in ("bin", "sbin", "lib", "lib64", "lib32", "libx32"):  
        p = "/" + d
        if os.path.islink(p):
            args += ["--symlink", os.readlink(p), p]
        elif os.path.isdir(p):
            args += ["--ro-bind", p, p]
    for src, dst in extra_binds:
        args += ["--ro-bind", src, dst]
    if gpu is not None:
        args += ["--ro-bind", "/sys", "/sys"]
    if venv:
        args += ["--ro-bind", os.path.abspath(venv), VENV_PATH]
    for k, v in (env or {}).items():
        args += ["--setenv", k, str(v)]
    args += ["--proc", "/proc", "--dev", "/dev"]
    if tmp:
        args += ["--tmpfs", "/tmp", "--setenv", "TMPDIR", "/tmp"]
    for src, dst in writable:
        args += ["--bind", os.path.abspath(src), dst]
    for dev in gpu_devices(gpu) if gpu is not None else []:
        args += ["--dev-bind", dev, dev]
    args += ["--remount-ro", "/",            # the sandbox's own root: nothing else writable anywhere
             "--chdir", chdir]
    return args

_warned_local = False
def _warn_local():
    """
    tinyagents runs behind two layers of protection
        1. the guard rail (an in process check on what the code is allowed to do)
        2. The code runs inside bubblewarp (bwrap) sandbox (setup by _bwrap_args)

    isolation="local" is the way around that, the agent code just runs as a plain 
    subprocess with your network and file access, only the guard rails still apply
    """
    global _warned_local
    if not _warned_local:
        warnings.warn("tinyagents isolation='local': agent code runs WITHOUT bubblewrap, in a plain process "
                      "with your user's file and network access. Only the guard rail and resource limits "
                      "stand in its way.", stacklevel=3)
        _warned_local = True

@dataclass
class ExecResult:
    output: str
    error: str | None
    final: bool = False
    value: object = None

class Sandbox:
    """
    One persistent, isolated Python session (variables survive between run() calls)
    A step that hangs past its timeout gets the whole sandbox killed and restarted, 
    which looses the session and we let the agent know this happened

        tools: {name: Tool} callable from the session code, each call is run here
        on the host (implemented later in Tool) and only the result goes back in the 
        sandbox!
    """

    def __init__(self, 
                 imports=DEFAULT_IMPORTS, 
                 timeout: int = 15, 
                 memory_mb: int = 2048,
                 cpu_seconds: int = 600, 
                 max_output: int = 4000, 
                 check: bool = True,
                 tools: dict | None = None, 
                 venv: str | None = None, 
                 isolation: str = "bwrap"):

        assert isolation in ("bwrap", "local"), f"isolation must be 'bwrap' or 'local', not {isolation!r}"
        self.tools = tools or {}
        self.cfg = dict(imports=list(imports), memory_mb=memory_mb, cpu_seconds=cpu_seconds,
                        max_output=max_output, check=check, tools=list(self.tools))
        self.timeout = timeout
        self.venv = venv
        self.isolation = isolation
        self.proc = None
        self.stderr = deque(maxlen=50)
        self._start()

    def _start(self):
        if self.isolation == "local":
            _warn_local()
            python = os.path.join(self.venv, "bin", "python") if self.venv else sys.executable
            # command to run python
            # -I in isolated mode, leaves the current dir off sys.path
            # -B dont write any .pyc files, no need for pycache
            # -u unbuffered stdout/stderr, write immediately
            # os.path.abspath(__file__) rerun this file the worker code lives here in the same module
            # flag the entry point checks, as this script is being RERUN as a sandbox worker not a normal script
            cmd = [python, "-I", "-B", "-u", os.path.abspath(__file__), "--sandbox-worker"]
        else:
            cmd = _bwrap_args([(os.path.abspath(__file__), WORKER_PATH)], venv=self.venv) + \
                  [_sandbox_python(self.venv), "-I", "-B", "-u", WORKER_PATH, "--sandbox-worker"]

        # the worker keeps running holding the agent's python session across steps, 
        # the PIPE allows the parent to talk to the worker only through them
        self.proc = subprocess.Popen( # spawn new process but nonblocking
            cmd, 
            stdin=subprocess.PIPE, # allow us to send inputs
            stdout=subprocess.PIPE, # allow us to capture the outputs
            stderr=subprocess.PIPE, # allow us to capture the errors
            text=True, # we want to send text NOT bytes
            bufsize=1 # line buffering, data stays in memory until new line \n
        )

        # drain the stderr in the background. Pipes only hold 64Kb of data by default, 
        # so if you dont drain as data becomes available, the worker will block. So 
        # the _drain_stderr keeps reading from proc (the reading is what drains it)
        # and adding to our deque. I learned about this here: https://biriukov.dev/docs/fd-pipe-session-terminal/2-pipes/
        threading.Thread(target=self._drain_stderr, args=(self.proc,), daemon=True).start()

        # send the config as a json file, this holds our allowed imports, memory, cpu limits, max output size
        # guard rail settings and tool names. The worker will then read it, set itself up, and create a stub function for 
        # each of the tools. When the agent calls a stub, the call is passed back and run on the host!
        self._send(self.cfg)

        # wait for the hello, it means its ready
        hello = self._recv(15)
        if not hello or not hello.get("ready"):
            self.close()
            raise RuntimeError("sandbox failed to start: " + " | ".join(self.stderr))
        
    def _drain_stderr(self, proc):
        for line in proc.stderr:
            self.stderr.append(line.rstrip())

    def _send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n") # always end in newline as thats what its looking for
        self.proc.stdin.flush()  

    def _recv(self, timeout):

        # The select.select() function is a low-level operating system primitive used for I/O multiplexing. 
        # It monitors multiple file descriptors (such as sockets, pipes, or files) and blocks 
        # until one or more of them become ready, or if we hit timeout. We just care about rlist, is it ready
        # to be read?
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not ready:
            return None
        line = self.proc.stdout.readline() # it sends back through the pipe: {"ready": true, "python": "3.x.y"}
        return json.loads(line) if line else None

    def _call_tool(self, msg):
        """
        run a toolcall from the sandbox here on the host
        """
        t = self.tools.get(msg["tool"])

        # check if the tool exists
        if t is None:
            return {"error": f"no tool named {msg['tool']!r}"}
        try:
            # get the function arguments and bind our input args, kwargs. 
            # if something doenst match it was called incorrectly
            inspect.signature(t.fn).bind(*msg["args"], **msg["kwargs"])  # wrong arguments -> a clear error
        except TypeError as e:
            # .name / .signature are defined later in Tool
            return {"error": f"{t.name}{t.signature}: {e}"}

        # try to actually run the function
        try:
            result = t.fn(*msg["args"], **msg["kwargs"])
        except Exception as e:
            return {"error": f"{t.name} failed: {type(e).__name__}: {e}"}

        # try to see if the  can be json dumped, otherwise use repr
        try:
            json.dumps(result)
        except (TypeError, ValueError):
            result = repr(result)

        return {"result": result}

    def set_vars(self, values):
        """
        puts variables from the host into the agents python session, so agent code can use them 
        by name without them being written into the prompt. For example, lets say our agent 
        is doing:
            agent.run("summarize this", inputs={"doc": long_text}). 
        We will make doc a variable in teh session that contains this data, we then add a note
        to the task listing the name and types. The model learns the variable exists, but the long text
        itself never goes into the context of the task prompt. Later the model can read it by
        writing code like:
            
        Thought: let me see how long it is and how it starts.
            <code>
            print(len(doc))
            print(doc[:3000])
            </code>
        
        """
        self._send({"vars": values})
        self._recv(15) # read the {"ok": true}, if we dont then that reply is unread and remains in buffer
    
    def run(self, code):
        try:
            self._send({"code": code, "timeout": self.timeout})
            while True: # wait loop

                # the workers main alarm is self.timout, it raises inside teh agent code and then
                # sends back, we just add a bufer so we can capture that
                resp = self._recv(self.timeout + 5)

                # There are two kinds of responses we can have. Either a tool call and the worker is blocked
                # waiting for an answer, or the host runs the real function with _call_tool, sends back the
                # result and waits more, as one step can make many tool calls
                # Or the final result which isnt a tool call (no tool key)..If the resp is none it means it
                # timed out, nothing arrived in time, stdout closed.  
                if resp is None or "tool" not in resp:
                    break

                # run that tool send it back
                self._send(self._call_tool(resp))

        except (BrokenPipeError, OSError):
            # if the worker has already died, writing to its stdin raises a brokenpipe error
            resp = None

        if resp is None:
            died = self.proc.poll() is not None # if it died it will return an exit code, if its running its None
            self.close() # kill the old worker, startup a new one
            self._start()
            reason = "the sandbox process died (memory or CPU limit?)" if died else \
                f"the step ran past {self.timeout}s and was killed"
            return ExecResult("", f"{reason}; the session was restarted, so earlier variables are gone")
        
        return ExecResult(resp["output"], resp["error"], resp["final"], resp["value"])

    def close(self):
        proc = getattr(self, "proc", None)
        if proc and proc.poll() is None:
            proc.kill() # end it
            proc.wait() # wait till it terminates (blocking)

    def __enter__(self):
        """this is our context mananger"""
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

def run_isolated(program: str, 
                 timeout: float = 20, 
                 memory_mb: int = 2048, 
                 venv: str | None = None,
                 isolation: str = "bwrap", 
                 allow_unsandboxed: bool = False, 
                 tmp: bool = False,
                 nproc: int = 1, 
                 env: dict | None = None):
    """
    Unlike sandbox that keeps one long-lived python session that the agent sends code
    blocks to, run_isolated is a one-off, run one complete program just once in a fresh
    process, exit, and return the output. 

    We will mostly use this for benchmark test runners, for example HumanEval has its own
    solution together with tests, we will use this to just check if it passes
    """

    fsize = (256 << 20) if tmp else 0 # 256 MB of space in bytes

    # max number of file descriptors. 32 is plenty but some things like BigCodeBench writes
    # and opens files, so they are given more room (they will write to tmp)
    nofile = 256 if tmp else 32 

    # F string that is python code to set limits inside the child. Earlier we set this in the 
    # worker, we now just set it here as its a one off
    prologue = f"""\
import resource as _r

for _k, _v in (
    (_r.RLIMIT_NPROC, {nproc}),
    (_r.RLIMIT_AS, {memory_mb << 20}),
    (_r.RLIMIT_CPU, {int(timeout) + 1}),
    (_r.RLIMIT_FSIZE, {fsize}),
    (_r.RLIMIT_NOFILE, {nofile}),
):
    _r.setrlimit(_k, (_v, _v))

del _r, _k, _v
""" 
    # if running locally
    if isolation == "local":
        if not allow_unsandboxed:
            raise RuntimeError("run_isolated runs code with no guard rail; without bubblewrap it would run "
                               "unsandboxed. Pass allow_unsandboxed=True if you really want that.")
        _warn_local()
        cmd = [os.path.join(venv, "bin", "python") if venv else sys.executable, "-I", "-B", "-"]

    # or run it sandboxed
    else:
        cmd = _bwrap_args(venv=venv, tmp=tmp, env=env, chdir="/tmp" if tmp else "/") + \
              [_sandbox_python(venv), "-I", "-B", "-"]

    # run with the prologue first
    try:
        r = subprocess.run(cmd, input=prologue + program, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        return -1, e.stdout or "", "timed out"

@dataclass
class Tool:
    """
    A function the agent can call from its code, it runs here on the host, so it
    can do what the sandbox cant (internet, files, another agent, etc...). The
    agent sees it in its system prompt as a python stub

        def web_search(query: str, max_results: int = 5) -> str:
            '''Searches the web ...'''

        
    to_prompt just makes the string that goes into the model look like the function we type
    """
    # @tool
    # def web_search(query: str, max_results: int = 5) -> str:
    #     """Searches the web and returns the top results.
    #
    #     Args:
    #         query: what to search for
    #         max_results: how many results to return
    #     """
    #     return "..."
    #
    # def web_search(query: str, max_results: int = 5) -> str:
    #     """
    #     Searches the web and returns the top results.
    #
    #     Args:
    #         query: what to search for
    #         max_results: how many results to return
    #     """
    fn: Callable
    name: str
    description: str
    signature: str

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def to_prompt(self):
        # indent the docstring 
        doc = "\n".join("    " + line if line.strip() else "" for line in self.description.splitlines())
        return f'def {self.name}{self.signature}:\n    """\n{doc}\n    """'

def tool(fn: Callable = None, *, name: str | None = None):
    """
    Turn a typed, documented function into a Tool:

        @tool
        def square_root(x: float) -> float:
            '''Returns the square root of a number.

            Args:
                x: the number
            '''
            return math.sqrt(x)

    or with a different name for the model to see:

        @tool(name="search")
        def _ddg_search_impl(query: str) -> str: ...

    @tool(name=...) calls tool() before it has the function, so fn is None:
    it returns the real decorator (lambda f: tool(f, name=name)), which Python
    then calls with the function. name is keyword-only so @tool("search")
    is an error instead of "search" landing in fn.
    """

    # if the function is None
    if fn is None:
        return lambda f: tool(f, name=name)

    # get the docstring
    doc = inspect.getdoc(fn)
    assert doc, f"tool {fn.__name__} needs a docstring: the agent only knows what it reads there"

    # get the function signature
    sig = inspect.signature(fn)

    # if there are typehints missing
    missing = [p for p in sig.parameters if sig.parameters[p].annotation is inspect.Parameter.empty]
    assert not missing, f"tool {fn.__name__}: add type hints for {missing}"

    # build the tool
    return Tool(fn, name or fn.__name__, doc, str(sig))

@dataclass
class Completion:
    text: str
    finish_reason: str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    ttft: float | None = None
    latency: float = 0.0

class OpenAIModel:
    """Any OpenAI-compatible chat endpoint (tinyserve, vLLM, ...), always streamed."""

    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1", api_key: str = "unused",
                 max_tokens: int = 2048, temperature: float | None = None, extra_body: dict | None = None,
                 retries: int = 4):

        # host side only, the sandbox worker never imports this as it wont have it installed
        from openai import OpenAI  

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model, self.max_tokens, self.temperature, self.extra_body = model, max_tokens, temperature, extra_body
        self.retries = retries

    def generate(self, messages: list[dict], stop: list[str], on_token=None) -> Completion:
        """
        One streamed completion. A call that fails before its first token (server
        restarting, overloaded, connection dropped) is retried with backoff; one that
        fails midway is not, since its tokens were already handed out
        """
        import openai

        transient = (
            openai.APIConnectionError, 
            openai.APITimeoutError, 
            openai.InternalServerError,
            openai.RateLimitError
        )

        for attempt in range(self.retries + 1):
            got_tokens = []
            try:
                return self._generate(messages, stop, on_token, got_tokens)
            except transient:
                if got_tokens or attempt == self.retries:
                    raise
                time.sleep(min(2 ** attempt, 30))

    def _generate(self, messages, stop, on_token, got_tokens) -> Completion:
        t = time.perf_counter()
        kwargs = dict(model=self.model, messages=messages, max_tokens=self.max_tokens, stop=stop,
                      stream=True, stream_options={"include_usage": True})
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        parts, finish, usage, ttft = got_tokens, None, None, None
        for chunk in self.client.chat.completions.create(**kwargs):
            if chunk.usage:
                usage = chunk.usage
            if chunk.choices:
                c = chunk.choices[0]
                if c.delta and c.delta.content:
                    if ttft is None:
                        ttft = time.perf_counter() - t
                    parts.append(c.delta.content)
                    if on_token:
                        on_token(c.delta.content)
                finish = c.finish_reason or finish
        details = getattr(usage, "prompt_tokens_details", None)
        return Completion("".join(parts), 
                          finish,
                          getattr(usage, "prompt_tokens", 0) or 0, 
                          getattr(usage, "completion_tokens", 0) or 0,
                          getattr(details, "cached_tokens", 0) or 0, 
                          ttft, 
                          time.perf_counter() - t)

@dataclass
class Step:
    number: int
    model_output: str
    code: str | None
    observation: str
    error: str | None
    final: bool
    completion: Completion | None = None

@dataclass
class RunResult:
    output: object
    state: str
    steps: list[Step]
    plans: list[str]
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    seconds: float

class MaxStepsReached(Exception):
    pass

class AgentInterrupted(Exception):
    pass

def parse_code(text):
    """
    Extract the codeblock
    """
    m = re.search(r"<code>\s*\n?(.*?)(?:</code>|$)", text, flags=re.S)
    if m and m.group(1).strip():
        return m.group(1).rstrip()
    m = re.search(r"```(?:python|py)?\s*\n(.*?)(?:```|$)", text, flags=re.S)
    if m and m.group(1).strip():
        return m.group(1).rstrip()
    # no block at all, but a bare final_answer(...) line: models often finish like that
    m = re.search(r"^[ \t]*(final_answer\(.*\))[ \t]*$", text, flags=re.M)
    return m.group(1) if m else None

_terminal = None
def terminal_view():
    """the shared terminal printer for verbose agents (one, so nested agents print in order)"""
    global _terminal
    if _terminal is None:
        try:
            from terminal import TerminalView  # rich, plain printing if it's not installed
            _terminal = TerminalView()
        except ImportError:
            _terminal = print_events
    return _terminal

def print_events(m: dict):
    """plain-text on_event (when rich isn't installed): the model's text as it streams, then each observation"""
    if m["type"] == "plan":
        print(f"\n--- plan ---\n{m['plan']}\n", flush=True)
    elif m["type"] == "llm_start" and m.get("step") is not None:  # (planning calls have no step)
        print(f"\n--- step {m['step']} ---", flush=True)
    elif m["type"] == "delta":
        print(m["text"], end="", flush=True)
    elif m["type"] == "step":
        print(f"\nObservation:\n{m['observation'] or '(no output)'}", flush=True)
    elif m["type"] == "final":
        print(f"\n--- final answer ---\n{m['answer']}", flush=True)

# the agents run on this thread, outermost first. A sub agents whole run happens inside
# the managers tool call. 
_running = threading.local()

class CodeAgent:
    """
    tools: what tools are available to the agent
    agents: other CodeAgents with a name and description, that this agent can hand tasks to
    planning_interval: write a plan before step 1 and update it ever N steps
    final_answer_checks: functions check(answer, agent) run on every final_answer(), returning False
                         or a message (or raising) rejects it , and the agent is told why and keeps going
    instruction: extra text appended to the system prompt
    name, description: what a manager agent sees when this agent is in its agents=[...]
    venv, memory_mb: this sessions python environment and memory cap
    confirm: a human in the loop, we can inspect code before every codeblock runs
    text_answers: a reply with no codeblocks is the final answer (for chat purposes where the model just talks)
    verbose: prints the runs in termainal as it happens (uses claude gen terminal.py)

    """

    def __init__(
            self, 
            model: OpenAIModel, 
            tools=(), 
            agents=(), 
            imports=DEFAULT_IMPORTS, 
            max_steps: int = 6,
            on_event=None, 
            step_timeout: int = 15, 
            max_output: int = 4000,
            system_prompt: str = SYSTEM_PROMPT, 
            instructions: str | None = None,
            planning_interval: int | None = None, 
            final_answer_checks=(),
            name: str | None = None, 
            description: str | None = None,
            venv: str | None = None, 
            memory_mb: int = 2048, 
            isolation: str = "bwrap",
            confirm: Callable | None = None, 
            text_answers: bool = False, 
            verbose: bool = True
        ):

        self.model = model
        self.imports = list(imports)
        self.max_steps = max_steps
        self.memory_mb = memory_mb
        self.step_timeout = step_timeout
        self.max_output = max_output
        self.venv = venv
        self.isolation = isolation
        self.planning_interval = planning_interval
        self.final_answer_checks = list(final_answer_checks)
        self.name = name
        self.description = description
        self.confirm = confirm
        self.text_answers = text_answers

        # the caller's event callback, or one that does nothing, so _emit never has to check for None
        user_events = on_event or (lambda m: None)

        # Verbose, every agent goes to the shared terminal printer and the callers callback
        self.on_event = (lambda m: (terminal_view()(m), user_events(m))) if verbose else user_events

        self.tools = {t.name: t for t in tools}
        self.agents = list(agents)

        # loop through any available sub agents
        for sub in agents:
            assert sub.name and sub.description, "an agent in agents=[...] needs a name and a description"
            self.tools[sub.name] = sub.as_tool() # this agent can be used via toolcalls

        # tools as prompt text
        tools_note, tools_text = "", ""
        if self.tools:
            # "There is no file, network or operating-system access" +  " except through the functions listed below"
            # otherwise we could give a websearch function, but then also say there is no network which is confusing
            tools_note = " except through the functions listed below"

            # add it to our tools section
            tools_text = TOOLS_SECTION.format(stubs="\n\n".join(t.to_prompt() for t in self.tools.values()))

            # if we have sub agents then we let the model know that any tool that has some text about it
            # being a Team Member is a self-contained agent and how to provide information to them
            if agents:
                tools_text += TEAM_NOTE

        self.system_prompt = system_prompt.format(
            imports=", ".join(self.imports), 
            tools_note=tools_note, 
            tools=tools_text, 
            instructions=(f"\n\n{instructions}" if instructions else "") + (TEXT_ANSWER_NOTE if text_answers else "")
        )

        self.steps: list[Step] = []
        self.plans: list[str] = []
        self.messages: list[dict] = []
        self.sandbox: Sandbox | None = None
        self.result: RunResult | None = None
        self._interrupted = False
            
    def as_tool(self):
        """this agent as a function another agent can call: name(task) -> answer"""
        def call(task):
            return str(self.run(task))
        doc = (f"Team member: {self.description}\n\nArgs:\n    task: a complete, self-contained description "
               f"of what to do and what to return")
        return Tool(call, self.name, doc, "(task: str) -> str")

    def _emit(self, m):
        """
        send one event, labeled with this agents name and nesting depth
        """
        # the agents running on this thread, outermost first (pushed/popped by run())
        stack = getattr(_running, "stack", [])

        # position on the stack (depth = 0 means top agent, 1 for its subagent, etc...)
        depth = stack.index(self) if self in stack else 0

        # new dict so the caller's m isnt modified, unnamed agents show up as "agent"
        # we copy key from m and then just add some extra metadat of where it came from
        self.on_event({**m, "agent": self.name or "agent", "depth": depth})

    def interrupt(self):
        """
        stop everything and the cooresponding sub-agents
        """
        self._interrupted = True
        for sub in self.agents:
            sub.interrupt()

    def _new_sandbox(self):
            self.close() # santiy check, close before opening a new one
            tools = {
                n: Tool(self._announce(t), t.name, t.description, t.signature) for n, t in self.tools.items()
            }
            self.sandbox = Sandbox(self.imports, timeout=self.step_timeout, max_output=self.max_output,
                                   tools=tools, venv=self.venv, memory_mb=self.memory_mb, isolation=self.isolation)

    def close(self):
        if getattr(self, "sandbox", None) is not None:
            self.sandbox.close()
            self.sandbox = None
            
    def _announce(self, t: Tool):
        """
        A simple wrapper that announces every tool call when it happens so we can log them
        """
        def fn(*args, **kwargs):
            # the wrapper accepts any argumnents and passes them on unchanged
            shown = [repr(x) for x in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
            # rebuilds the arguments the way theyd look in a call, first positional args then key/value pairs
            self._emit({"type": "tool", "name": t.name, "args": truncate(", ".join(shown), 200)})
            # calls the actual tool and returns results unchanged
            return t.fn(*args, **kwargs)
        # make sure the function signature doesnt change as the earlier inspect.signature(t.fn).bind(*msg["args"], **msg["kwargs"])
        # would otherwise not match!
        fn.__signature__ = inspect.signature(t.fn)

        return fn

    def _llm(self, messages, stop, step=None):
        """
        One model call with events around it and its usage recorded
        """
        # step is None for planning and the final max-steps call
        self._emit({"type": "llm_start", "step": step})

        # stream the next token of text
        c = self.model.generate(messages, stop, lambda t: self._emit({"type": "delta", "text": t}))

        # Timing and token counts for this call
        self._emit(
            {
                "type": "llm_end", 
                "latency": c.latency, 
                "ttft": c.ttft, 
                "prompt": c.prompt_tokens,
                "completion": c.completion_tokens, 
                "cached": c.cached_tokens
            }
        )

        self._completions.append(c) # _completions defined in _run later

        return c


    def _plan(self, task, first, remaining):
        """
        A planning step, its own call with the same system prompt so we get still get our
        prefix cache hit on this step
        """

        if first:
            # First plan of this run is our system promot + "survey the facts and plan"
            # self.messages[0] is the same system prompt as every other call so the prefix
            # cache can reuse it. 
            ask = [self.messages[0], {"role": "user", "content": PLAN_PROMPT.format(task=task)}]
        else:
            # The whole conversation so far + "update your plan, you have N steps left"
            ask = self.messages + [{"role": "user", "content": UPDATE_PLAN_PROMPT.format(remaining=remaining)}]

        # the model then writes the plan and we want to end on the <end_plan> token which is our stop string
        plan = self._llm(ask, ["<end_plan>"]).text.strip()

        # Kept for RunResult, save, replay and shown as a "plan" event
        self.plans.append(plan)
        self._emit({"type": "plan", "plan": plan})

        # The plan still goes into the real history, then a user turn is twhat sends it back to work, 
        # on the other hand the planning prompt itself isnt added to the history, only the main
        # system prompt gets its cache hits
        self.messages += [{"role": "assistant", "content": f"Here is my plan:\n{plan}"},
                          {"role": "user", "content": "Now carry out the plan, one step at a time."}]

    def _check_answer(self, value):
        """
        runs our final checks. For example we could say our final answer must be positive with:
            lambda answer, agent: None if a > 0 else "must be positive"
        Then if our value is negative, our lambda catches it and gives a text reason back
        """
        for check in self.final_answer_checks:
            try:
                verdict = check(value, self)
            except Exception as e:
                verdict = f"{type(e).__name__}: {e}"
            # Rejection = False or a string with the reason, True is None or anything else = accepted
            if verdict is False or isinstance(verdict, str):
                return verdict if isinstance(verdict, str) else f"{check.__name__} rejected it"
        return None

    def run(self, task: str, inputs: dict | None = None, reset: bool = True, max_steps: int | None = None):

        # _running is the local thread started up ealier, we push this agent onto the stack. Each thread
        # has its own stack, a list of agents currently running on that thread. This is just used in the 
        # terminal logging, here is an example:
        # manager.run("write a report")          stack = [manager]               depth 0
        #   code calls researcher("find X")      <- tool call on the host, same thread
        #     researcher.run("find X")           stack = [manager, researcher]   depth 1
        #     ...researcher's steps print indented...
        #     return                             stack = [manager]
        # manager continues
        _running.stack = getattr(_running, "stack", []) + [self]

        try:
            # run the task
            return self._run(task, inputs, reset, max_steps)
        finally:
            # once done remove it from the stack
            _running.stack = _running.stack[:-1]

    def _run(self, task, inputs, reset, max_steps):
        """
        The main agent loop that can either continue or start over, for example:

            agent = CodeAgent(model)
            agent.run("load these numbers and add", inputs={"nums": [2,1})   # run 1
            agent.run("now the average", reset=False)                        # run 2: continues run 1
            agent.run("what is 2**10?")                                      # run 3: fresh start
        """

        # start the clock, we can also set a per run steps to override defaults
        t0 = time.perf_counter()
        max_steps = max_steps or self.max_steps 
        
        self._interrupted = False
        self._completions = []

        # a fresh start or the very first run
        if reset or self.sandbox is None:
            self._new_sandbox() 
            self.steps = []
            self.plans = []
            self.messages = [{"role": "system", "content": self.system_prompt}]

        if inputs:
            # put the variables into the session (sent as JSON, so plain data only)
            self.sandbox.set_vars(inputs)
            # and tell the model they exist, by name and type only (not their contents:
            # the model reads them with code, e.g. print(doc[:2000]))
            task += "\n\nThese variables are already defined in your Python session: " + \
                    ", ".join(f"{k} ({type(v).__name__})" for k, v in inputs.items())

        # the task as a user message: "Task: ..." for a fresh run, "New task: ..." for a follow-up
        self.messages.append({"role": "user", "content": f"{'Task' if reset else 'New task'}: {task}"})
        self._emit({"type": "run_start", "task": task})

        # step numbers continue across followups, a second run after 4 steps starts at 5
        first_step = len(self.steps) + 1

        # How the run ends, unless the loop says otherwise
        state = "max_steps"
        answer = None

        for n in range(first_step, first_step + max_steps):

            # A stop request is checked before every model call
            if self._interrupted:
                state = "interrupted"
                break

            # steps already taken in this run
            done = n - first_step

            # with planning on, a first plan before the first step, then an update
            # every planning interval steps
            if self.planning_interval and done % self.planning_interval == 0:
                self._plan(task, first=(done==0), remaining=max_steps - done)

            # The model writes "Thought: ... <code> ... </code>" and then an Observation:
            # of the run code is provided back like this:
            #
            # <code>
            # print(fib(1), fib(2), fib(10))
            # </code>
            # Observation: 1 1 55
            # Thought: fib(10) is 55.
            # Now if the model starts generating Observation:, it will also make up the
            # actual outputs (which our system is supposed to run). The model may not
            # emit </code> and start to hallucinate, it may use the python fence, 
            # it may write no code at all. In each case Observation: is our secondary
            # catch. 
            c = self._llm(self.messages, STOP, step=n)

            # get the reply
            reply = c.text.rstrip()

            # parse the code block (None if there isnt one)
            code = parse_code(reply)

            # the stop string would cut off "</code>" as they are not
            # included in the output, we just add it back so the
            # history is correct
            if code is not None and "<code>" in reply and "</code>" not in reply:
                reply += "\n</code>"

            # the reply joins the history
            self.messages.append({"role": "assistant", "content": reply})

            # The though is everything before the codeblock (whole reply if there is no code)
            thought = re.split(r"<code>|```", reply, maxsplit=1)[0].strip()

            # a plain text final answer, no code, text answers allowed and something was said
            text_answer = code is None and self.text_answers and thought

            # A text answer is shown once as the final answer, not also as a thought
            self._emit({"type": "action", "step": n, "thought": "" if text_answer else thought, "code": code})

            # Decide what to do with the reply
            declined = None
            # if we have code, we havent interrupted and we want human confirmation
            if code is not None and not self._interrupted and self.confirm:
                declined = self.confirm(code, self)

            # if we want text only answers ExecResults(output, error, final, value)
            if text_answer:
                # plain text is the answer: a "final" result without running anything,
                # with a leading "Thought:" label removed from the answer
                result = ExecResult("", None, final=True, value=re.sub(r"^Thought:\s*", "", thought))

            elif code is None:
                # if there was no code found nothing to do
                result = ExecResult("","no code found. Reply with 'Thought: ...' then a <code>...</code> block.")

            elif self._interrupted:
                # interrupted while the model was writing
                result = ExecResult("(not run: the user stopped the run)", None)

            elif declined is not None:
                # human said not to run this block, and why
                result = ExecResult(f"(not run: the user declined this code. Their note: {declined or 'none'})", None)

            else:
                # run it!
                result = self.sandbox.run(code)

            # a final answer must get past the check
            if result.final:
                rejected = self._check_answer(result.value)
                if rejected:
                    # Replace the result with a nonfinal one if rejected
                    result = ExecResult(result.output, f"final_answer({result.value!r:.200}) was rejected: "
                                                       f"{rejected}. Fix it and call final_answer again.")

            # build the observation the model will see next
            observation = result.output.rstrip()
            if result.error:
                # the error goes after the output (newline only if there was output) plus a hint
                # to try to take a different approach
                observation = (observation + "\n" if observation else "") + f"Error: {result.error}{RETRY_HINT}"

            # record the step, the raw model text, the code, waht came back and the completion
            self.steps.append(
                Step(
                    n, c.text, code, observation, result.error, result.final, c
                )
            )

            # the event will cary only the last 1500 characters for succinctness
            self._emit(
                {
                    "type": "step", 
                    "step": n, 
                    "error": result.error is not None,
                    "final": result.final, 
                    "observation": observation[-1500:]
                }
            )

            # done or feed back and go again
            if result.final:
                state, answer = "success", result.value
                break

            # the observation as the next user message
            self.messages.append({"role": "user", "content": f"Observation:\n{observation or '(no output)'}"})

        if state == "max_steps": # if we hit our max, one last call to get the best answer we can
            ask = self.messages + [{"role": "user", "content": MAX_STEPS_PROMPT.format(task=task)}]
            answer = self._llm(ask, stop=[]).text.strip()

        # wrapup total for this run
        cs = self._completions
        self.result = r = RunResult(
            answer, state, list(self.steps), list(self.plans), 
            sum(c.prompt_tokens for c in cs), sum(c.completion_tokens for c in cs),
            sum(c.cached_tokens for c in cs), time.perf_counter() - t0
        )

        # final event only the last 2000 characters print
        self._emit({"type": "final", "answer": str(answer)[:2000], "state": state, "steps": len(r.steps),
                    "seconds": r.seconds, "prompt": r.prompt_tokens, "cached": r.cached_tokens})

        # an interrupted run raises so callers cant mistake its None or a real one
        if state == "interrupted":
            raise AgentInterrupted(f"interrupted after {len(self.steps)} steps")

        return answer

    def return_full_code(self):
        """every code block the agent ran, as one script"""
        return "\n\n".join(s.code for s in self.steps if s.code)
    
    def save(self, path: str):
        """the agent's conversation (its trajectory) and steps as JSON"""
        steps = [dict(number=s.number, model_output=s.model_output, code=s.code, observation=s.observation,
                      error=s.error, final=s.final) for s in self.steps]
        r = self.result
        result = None if r is None else dict(output=str(r.output), state=r.state, seconds=round(r.seconds, 1),
                                             prompt_tokens=r.prompt_tokens, cached_tokens=r.cached_tokens,
                                             completion_tokens=r.completion_tokens)
        with open(path, "w") as f:
            json.dump(dict(name=self.name, model=self.model.model, tools=list(self.tools), messages=self.messages,
                           plans=self.plans, steps=steps, result=result), f, indent=2)

    def replay(self):
        """print the run back: each step's model output and observation"""
        for p in self.plans:
            print(f"--- plan ---\n{p}\n")
        for s in self.steps:
            print(f"--- step {s.number} ---\n{s.model_output.strip()}\nObservation:\n{s.observation or '(no output)'}\n")
        if self.result:
            print(f"--- {self.result.state}: {self.result.output!r:.500}")

    def __enter__(self):
        """
        this is so we can do with CodeAgent(...) as agent later, worker stops when the block ends
        """
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


if __name__ == "__main__" and sys.argv[1:] == ["--sandbox-worker"]:
    _worker_main()

# if __name__ == "__main__":

#     code = """
# import pandas
# import numpy as np
# from collections import Counter

# def foo(hi):
#     return hi * 2

# x = foo(10)
# print(x)
# """

#     # how check_code works
#     imports = {"pandas", "numpy", "torch"} # accepted imports
#     reserved = {"final_answer", "search"} # function names we dont want to overwrite
#     check_code(code, imports, reserved)

#     # how execute works
#     cfg = {
#             "check": True,     # configuration that we will load later
#             "imports": ["pandas", "numpy", "collections"], 
#             "tools": {}, 
#             "max_output": 10000
#         }
#     output = [] # where all of our prints get catptured and returned to the llm
#     namespace = {"__builtins__": safe_builtins(cfg["imports"], output)}
#     result = _execute(code, namespace, output, cfg, timeout=10)
#     print(result)

#     reply = """
# Here is the solution:

# <code>
# def add(a, b):
#     return a + b
# </code>

# This function adds two numbers.
# """
#     print(parse_code(reply))
    

#     reply = """
# Here is the solution:

# ```python
# def add(a, b):
#     return a + b
# ```

# This function adds two numbers.
# """
#     print(parse_code(reply))

#     reply = """
# The result is:

# final_answer(42)
# """
#     print(parse_code(reply))

 