
import ast

source = """
import pandas
import numpy as np
from collections import Counter
from os.path import join

def foo(a):
    return a + 2

x = foo(10)
y = bar(x, 20)
print(y)
"""


def describe_node(node):
    """Convert an AST node into a readable Python-like representation."""
    return ast.unparse(node)


try:
    tree = ast.parse(source)
except SyntaxError as e:
    print(f"SyntaxError: {e.msg} (line {e.lineno})")
    raise


print(ast.dump(tree, indent=4))

for node in ast.walk(tree):

    if isinstance(node, ast.Import):
        for alias in node.names:
            print(
                f"Import: {alias.name}, "
                f"alias: {alias.asname}, "
                f"line: {node.lineno}"
            )

    elif isinstance(node, ast.ImportFrom):
        for alias in node.names:
            print(
                f"From: {node.module}, "
                f"Import: {alias.name}, "
                f"alias: {alias.asname}, "
                f"line: {node.lineno}"
            )

    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args

        positional_args = [
            arg.arg for arg in args.posonlyargs + args.args
        ]

        keyword_only_args = [
            arg.arg for arg in args.kwonlyargs
        ]

        print(
            f"\nFunction: {node.name}\n"
            f"  Line: {node.lineno}\n"
            f"  Positional args: {positional_args}\n"
            f"  Keyword-only args: {keyword_only_args}\n"
            f"  *args: {args.vararg.arg if args.vararg else None}\n"
            f"  **kwargs: {args.kwarg.arg if args.kwarg else None}\n"
            f"  Defaults: {[describe_node(d) for d in args.defaults]}\n"
            f"  Return annotation: "
            f"{describe_node(node.returns) if node.returns else None}\n"
            f"  Decorators: {[describe_node(d) for d in node.decorator_list]}"
        )

    elif isinstance(node, ast.Call):

        # Determine the function being called.
        if isinstance(node.func, ast.Name):
            function_name = node.func.id

        elif isinstance(node.func, ast.Attribute):
            function_name = describe_node(node.func)

        else:
            function_name = describe_node(node.func)

        positional_args = [
            describe_node(arg) for arg in node.args
        ]

        keyword_args = {
            kw.arg: describe_node(kw.value)
            for kw in node.keywords
        }

        print(
            f"\nFunction call: {function_name}\n"
            f"  Line: {node.lineno}\n"
            f"  Positional args: {positional_args}\n"
            f"  Keyword args: {keyword_args}"
        )

    elif isinstance(node, ast.Assign):

        targets = [
            describe_node(target)
            for target in node.targets
        ]

        value = describe_node(node.value)

        print(
            f"\nAssignment:\n"
            f"  Line: {node.lineno}\n"
            f"  Targets: {targets}\n"
            f"  Value: {value}"
        )

    elif isinstance(node, ast.Return):
        value = (
            describe_node(node.value)
            if node.value is not None
            else None
        )

        print(
            f"\nReturn:\n"
            f"  Line: {node.lineno}\n"
            f"  Value: {value}"
        )

    elif isinstance(node, ast.Expr):
        print(
            f"\nExpression:\n"
            f"  Line: {node.lineno}\n"
            f"  Value: {ast.unparse(node.value)}\n"
            f"  Type: {type(node.value).__name__}"
        )
