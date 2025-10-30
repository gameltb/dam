import glob
import os
import sys
from pathlib import Path
from typing import List

import tomli
import contextlib
from poethepoet.app import PoeThePoet
from rich import print
import time
import threading


def _read_fd(fd: int) -> bytes:
    # Read all available bytes from an fd until EOF
    chunks: list[bytes] = []
    try:
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            chunks.append(chunk)
    except OSError:
        # If reading fails, return what we have
        pass
    return b"".join(chunks)


def run_callable_capture_fds(callable_obj) -> tuple[int, str, str]:
    """Run callable_obj() while capturing low-level FD 1 and 2 (stdout/stderr).

    Returns (exit_code, stdout_text, stderr_text).

    This captures output from subprocesses and C-level writes that Python-level
    redirect_stdout() won't catch.
    """
    # Flush Python-level buffers first
    sys.stdout.flush()
    sys.stderr.flush()

    read_out, write_out = os.pipe()
    read_err, write_err = os.pipe()

    # Save original fds
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)

    try:
        # Replace stdout/stderr with our write ends
        os.dup2(write_out, 1)
        os.dup2(write_err, 2)

        # Close original write fds in parent — the fd 1/2 now point to the pipe
        os.close(write_out)
        os.close(write_err)

        # Start background threads that will read from the pipe read-ends while
        # the callable runs. This prevents the pipe buffer filling up and
        # blocking writers (deadlock) when the callable produces lots of output.
        out_parts: list[bytes] = []
        err_parts: list[bytes] = []

        def _reader(fd: int, container: list[bytes]) -> None:
            try:
                data = _read_fd(fd)
                container.append(data)
            except Exception:
                # Best-effort: if reading fails, ignore and allow callable to run
                pass

        t_out = threading.Thread(target=_reader, args=(read_out, out_parts), daemon=True)
        t_err = threading.Thread(target=_reader, args=(read_err, err_parts), daemon=True)
        t_out.start()
        t_err.start()

        # Call the provided callable which may spawn subprocesses
        result = callable_obj()

        # Flush again to ensure data is written to the pipes
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

    finally:
        # Restore original stdout/stderr fds
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)

    # At this point we've restored the original stdout/stderr fds in the
    # finally block above; that operation closes the previous fd 1/2 which
    # were the pipe write-ends and will cause the reader threads to see EOF.
    # Wait for the threads to finish reading their data, then collect bytes.
    try:
        # t_out/t_err exist only if threads were started; join them if present
        t_out.join()
        t_err.join()
    except NameError:
        # Threads were not created (shouldn't happen), fall back to direct read
        out_bytes = _read_fd(read_out)
        err_bytes = _read_fd(read_err)
    else:
        out_bytes = b"".join(out_parts)
        err_bytes = b"".join(err_parts)

    os.close(read_out)
    os.close(read_err)

    out_text = out_bytes.decode(errors="replace")
    err_text = err_bytes.decode(errors="replace")

    return result, out_text, err_text


def discover_projects(workspace_pyproject_file: Path) -> List[Path]:
    with workspace_pyproject_file.open("rb") as f:
        data = tomli.load(f)

    projects = data["tool"]["uv"]["workspace"]["members"]
    exclude = data["tool"]["uv"]["workspace"].get("exclude", [])

    all_projects: List[Path] = []
    for project in projects:
        if "*" in project:
            globbed = glob.glob(str(project), root_dir=workspace_pyproject_file.parent)
            globbed_paths = [Path(p) for p in globbed]
            all_projects.extend(globbed_paths)
        else:
            all_projects.append(Path(project))

    for project in exclude:
        if "*" in project:
            globbed = glob.glob(str(project), root_dir=workspace_pyproject_file.parent)
            globbed_paths = [Path(p) for p in globbed]
            all_projects = [p for p in all_projects if p not in globbed_paths]
        else:
            all_projects = [p for p in all_projects if p != Path(project)]

    return all_projects


def extract_poe_tasks(file: Path) -> set[str]:
    with file.open("rb") as f:
        data = tomli.load(f)

    tasks = set(data.get("tool", {}).get("poe", {}).get("tasks", {}).keys())

    # Check if there is an include too
    include: str | None = data.get("tool", {}).get("poe", {}).get("include", None)
    if include:
        include_file = file.parent / include
        if include_file.exists():
            tasks = tasks.union(extract_poe_tasks(include_file))

    return tasks


def _print_task_summary(
    project: Path,
    task_name: str,
    task_args: List[str],
    out: str,
    err: str,
    result: int,
    duration: float,
) -> None:
    """Print stdout/stderr (if present) and a Finished task summary including duration.

    This helper centralizes duplicate printing logic used for both success (long-running)
    and failure cases.
    """
    if out:
        print(f"--- stdout for {('failed ' if result else '')}{task_name} in {project} ---")
        print(out)
    if err:
        print(f"--- stderr for {('failed ' if result else '')}{task_name} in {project} ---")
        print(err)
    print(
        f"Finished task {task_name} in {project} with args {task_args} with exit code {result} (took {duration:.2f}s)"
    )


def main() -> None:
    os.environ["MYPYPATH"] = "packages/"
    pyproject_file = Path(__file__).parent / "pyproject.toml"
    projects = discover_projects(pyproject_file)

    args = sys.argv[1:]

    package_to_run = None
    if "--package" in args:
        package_index = args.index("--package")
        if package_index + 1 < len(args):
            package_to_run = args[package_index + 1]
            # remove from args
            args.pop(package_index + 1)
            args.pop(package_index)
        else:  # --package is the last arg
            args.pop(package_index)

    if not package_to_run:  # Handles None and ""
        package_to_run = None

    # Separate script args from task args
    task_args: List[str] = []
    script_args: List[str]
    if "--" in args:
        separator_index = args.index("--")
        script_args = args[:separator_index]
        task_args = args[separator_index + 1 :]
    else:
        script_args = args

    if not script_args:
        print("Please provide a task name")
        sys.exit(1)

    task_name = script_args[0]

    if package_to_run:
        projects = [p for p in projects if p.name == package_to_run]
        if not projects:
            print(f"Package '{package_to_run}' not found in workspace.")
            sys.exit(1)

    # Construct the arguments to pass to poe
    poe_cli_args = [task_name]
    if task_args:
        poe_cli_args.append("--")
        poe_cli_args.extend(task_args)

    for project in projects:
        tasks = extract_poe_tasks(project / "pyproject.toml")
        if task_name in tasks:
            # print(f"Running task {task_name} in {project} with args {task_args}")
            app = PoeThePoet(cwd=project)
            # threshold for printing long-running task summary (seconds)
            notify_threshold = float(os.environ.get("RUN_TASK_NOTIFY_THRESHOLD", "8.0"))

            def _call(app=app) -> int:
                return app(cli_args=poe_cli_args)

            start = time.perf_counter()
            result, out, err = run_callable_capture_fds(_call)
            duration = time.perf_counter() - start

            # On failure: always print details and exit.
            if result:
                _print_task_summary(project, task_name, task_args, out, err, result, duration)
                sys.exit(result)

            # On success: only print summary when task ran longer than threshold
            if duration > notify_threshold:
                _print_task_summary(project, task_name, task_args, out, err, result, duration)
        else:
            # This is not an error, some packages might not have all tasks
            pass


if __name__ == "__main__":
    main()
