"""Execute this capstone's Python notebook cells in order and retain outputs.

This runner uses one Python namespace and supports the notebook's plain Python
cells and dataframe HTML output. It does not require a Jupyter server or kernel.
"""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys


def execute_notebook(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    original_cwd = Path.cwd()
    count = 0
    try:
        os.chdir(path.parent)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            count += 1
            source = cell["source"]
            if isinstance(source, list):
                source = "".join(source)
            print(f"Running cell {count}...", flush=True)
            tree = ast.parse(source)
            final = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                final = ast.Expression(tree.body.pop().value)
            stdout, stderr = io.StringIO(), io.StringIO()
            result = None
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(tree, f"{path.name}:cell-{count}", "exec"), namespace)
                if final is not None:
                    result = eval(compile(final, f"{path.name}:cell-{count}", "eval"), namespace)
            cell["execution_count"] = count
            cell["outputs"] = []
            for name, stream in [("stdout", stdout), ("stderr", stderr)]:
                if stream.getvalue():
                    cell["outputs"].append({"output_type": "stream", "name": name,
                                            "text": stream.getvalue()})
                    print(stream.getvalue(), end="", flush=True)
            if result is not None:
                data = {"text/plain": repr(result)}
                if hasattr(result, "_repr_html_"):
                    html = result._repr_html_()
                    if html is not None:
                        data["text/html"] = html
                cell["outputs"].append({"output_type": "execute_result", "execution_count": count,
                                        "data": data, "metadata": {}})
                print(repr(result), flush=True)
        notebook.setdefault("metadata", {})["language_info"] = {
            "name": "python", "version": sys.version.split()[0],
        }
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        if "con" in namespace:
            namespace["con"].close()
        print(f"Saved {count} executed cells to {path.name}.", flush=True)
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    execute_notebook(repo / "work/notebooks/capstone.ipynb")
