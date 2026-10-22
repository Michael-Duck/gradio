"""

Contains the functions that run when `gradio` is called from the command line. Specifically, allows

$ gradio app.py, to run app.py in reload mode where any changes in the app.py file or Gradio library reloads the demo.
$ gradio app.py my_demo, to use variable names other than "demo"
$ gradio package.app, to run a module in reload mode, the same way `python -m package.app` runs it.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Annotated, NamedTuple

import typer
from rich import print

import gradio
from gradio import utils

reload_thread = threading.local()

# The name of a module, e.g. `package.app`, in the syntax that `python -m` takes.
MODULE_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


class _ModuleTarget(NamedTuple):
    """A module that the CLI can run, e.g. `package.app`."""

    # The file that `python -m <module>` runs.
    file: Path
    # The name of the module that file is executed as.
    module_name: str


def _handle_interrupt():
    """Handle interrupt signals and logout based on user preference"""

    if os.getenv("GRADIO_VIBE_MODE") and os.getenv("GRADIO_AUTO_LOGOUT") == "true":
        try:
            from huggingface_hub import logout

            logout()
            print("\n\nLogged out of Hugging Face")
        except Exception as e:
            print(f"\n\nError logging out of Hugging Face: {e}")

    sys.exit(0)


def _module_name(demo_path: Path) -> str | None:
    """The name of the module that the CLI argument refers to, if any.

    `gradio` accepts either the path of a script or the name of a module, e.g.
    `package.app`, which is run with `python -m package.app`. An existing file
    always wins, so scripts that are launched by path keep working.
    """
    target = str(demo_path)
    if any(sep and sep in target for sep in (os.sep, os.altsep)):
        # A path, not the name of a module.
        return None
    if target.endswith(".py"):
        # A script, not the name of a module.
        return None
    if not MODULE_NAME_PATTERN.fullmatch(target):
        return None
    if Path(target).is_file():
        # An existing file, e.g. `gradio app`, is launched by path.
        return None
    return target


def _module_target(module_name: str) -> _ModuleTarget | None:
    """The file that `python -m <module_name>` would run.

    Returns None if the module cannot be found, or if it is a package without
    a `__main__.py`, in which case `python -m` cannot run it either.
    """
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.submodule_search_locations is None:
        if spec.origin is None:
            return None
        # `python -m mymodule` runs `mymodule.py` as the module `mymodule`.
        return _ModuleTarget(file=Path(spec.origin), module_name=module_name)
    # `python -m package.app` runs the `__main__.py` of the package as the
    # module `package.app.__main__`.
    try:
        main_spec = importlib.util.find_spec(f"{module_name}.__main__")
    except (ImportError, ValueError):
        return None
    if main_spec is None or main_spec.origin is None:
        return None
    return _ModuleTarget(
        file=Path(main_spec.origin), module_name=f"{module_name}.__main__"
    )


def _setup_config(
    demo_path: Path,
    additional_watch_dirs: list[str] | None = None,
    watch_library: bool = False,
):
    # guarantee access to the module of an app
    sys.path.insert(0, os.getcwd())

    module_arg = _module_name(demo_path)
    if module_arg:
        # The app is a module, e.g. `package.app`. It is run with `python -m`,
        # so the file that is watched is the file that `python -m` runs.
        target = _module_target(module_arg)
        if target is None:
            raise FileNotFoundError(
                f"Could not run '{module_arg}' as a module. Pass the path of a "
                f"Python file, e.g. `gradio app.py`, or the name of a module that "
                f"can be run with `python -m {module_arg}`."
            )
        abs_original_path = target.file
        module_name = target.module_name
    else:
        original_path = Path(demo_path)
        abs_original_path = utils.abspath(original_path)

        if original_path.is_absolute():
            relpath = original_path.relative_to(Path.cwd())
        else:
            relpath = original_path
        module_name = str(relpath.parent / relpath.stem).replace(os.path.sep, ".")

    gradio_folder = Path(inspect.getfile(gradio)).parent

    message = "Watching:"
    message_change_count = 0

    watching_dirs = []
    if str(gradio_folder).strip() and watch_library:
        # This is a source install
        watching_dirs.append(gradio_folder)
        message += f" '{gradio_folder}'"
        message_change_count += 1

    abs_parent = abs_original_path.parent
    if str(abs_parent).strip():
        watching_dirs.append(abs_parent)
        if message_change_count == 1:
            message += ","
        message += f" '{abs_parent}'"

    abs_current = Path.cwd().absolute()
    if str(abs_current).strip() and abs_current not in watching_dirs:
        try:
            gradio_folder.relative_to(abs_current)
            is_subdir = True
        except ValueError:
            is_subdir = False
        if is_subdir and not watch_library:
            pass
        else:
            watching_dirs.append(abs_current)
            if message_change_count == 1:
                message += ","
            message += f" '{abs_current}'"

    for wd in additional_watch_dirs or []:
        if Path(wd) not in watching_dirs:
            watching_dirs.append(wd)

            if message_change_count == 1:
                message += ","
            message += f" '{wd}'"

    print(message + "\n")

    return module_name, abs_original_path, [str(s) for s in watching_dirs]


def main(
    demo_path: Path,
    demo_name: str = "",
    watch_dirs: list[str] | None = None,
    encoding: str = "utf-8",
    watch_library: bool = False,
    app_args: Annotated[
        list[str] | None,
        typer.Argument(
            metavar="[-- APP_ARGS...]",
            help="Arguments to pass to your app, placed after a `--` separator.",
        ),
    ] = None,
):
    """Run a Gradio app in reload mode.

    DEMO_PATH is either the path of a Python script or the name of a module,
    e.g. `package.app`, which is run the same way `python -m package.app` runs
    it.
    """
    signal.signal(signal.SIGINT, lambda _signum, _frame: _handle_interrupt())
    signal.signal(signal.SIGTERM, lambda _signum, _frame: _handle_interrupt())

    # A module is run with `python -m`, so that the app is imported as a module
    # and the relative imports of the package it lives in keep working.
    module_arg = _module_name(demo_path)

    # default execution pattern to start the server and watch changes
    try:
        module_name, path, watch_sources = _setup_config(
            demo_path, watch_dirs, watch_library
        )
    except FileNotFoundError as error:
        raise typer.BadParameter(str(error)) from error

    # Pass the following data as environment variables
    # so that we can set up reload mode correctly in the networking.py module
    env_vars = dict(
        os.environ,
        GRADIO_WATCH_DIRS=",".join(watch_sources),
        GRADIO_WATCH_MODULE_NAME=module_name,
        GRADIO_WATCH_DEMO_NAME=demo_name,
        GRADIO_WATCH_DEMO_PATH=str(path),
        GRADIO_WATCH_ENCODING=encoding,
    )

    if "GRADIO_VIBE_MODE" in os.environ:
        env_vars["GRADIO_VIBE_MODE"] = os.environ["GRADIO_VIBE_MODE"]

    if module_arg:
        # A module is run with `python -m`, so that the app is imported as a
        # module and the relative imports of the package it lives in keep
        # working.
        command = [sys.executable, "-u", "-m", module_arg]
    else:
        command = [sys.executable, "-u", path]

    # Everything after `--` belongs to the user's script, e.g.
    # `gradio app.py -- --name Gretel` for a script that uses argparse.
    popen = subprocess.Popen(
        [*command, *(app_args or [])],
        env=env_vars,
    )
    if popen.poll() is None:
        try:
            popen.wait()
        except (KeyboardInterrupt, OSError):
            pass


def build_app() -> typer.Typer:
    """Build the `gradio app.py [-- ...]` CLI."""
    app = typer.Typer(add_completion=False)
    app.command()(main)
    return app


def run() -> None:
    build_app()()


if __name__ == "__main__":
    run()
