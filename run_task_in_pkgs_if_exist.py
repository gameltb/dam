import glob
import sys
from pathlib import Path
from typing import List

import tomli
from poethepoet.app import PoeThePoet
from rich import print


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


import os


def main() -> None:
    os.environ["MYPYPATH"] = "packages/"
    pyproject_file = Path(__file__).parent / "pyproject.toml"
    projects = discover_projects(pyproject_file)

    args = sys.argv[1:]

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

    package_to_run = None
    if "--package" in script_args:
        try:
            package_index = script_args.index("--package")
            package_to_run = script_args[package_index + 1]
        except (ValueError, IndexError):
            print("Error: --package flag must be followed by a package name.")
            sys.exit(1)

    if task_name in ("check", "check-full", "check-ci"):
        if task_args:
            print(
                f"Warning: Passing extra arguments to the '{task_name}' task is not supported. They will be ignored."
            )
        # Map the umbrella check tasks to the sequence of poe subtasks
        tasks_to_run = {
            "check": ["fmt", "lint", "pyright", "test"],
            "check-full": ["fmt", "lint", "pyright", "mypy", "test"],
            "check-ci": ["fmt", "lint", "pyright", "test-cov"],
        }[task_name]

        # If a package was specified, filter the discovered projects now so
        # we can run the subtasks inside each project's PoeThePoet context.
        if package_to_run:
            projects = [p for p in projects if p.name == package_to_run]
            if not projects:
                print(f"Package '{package_to_run}' not found in workspace.")
                sys.exit(1)

        # Run each subtask using PoeThePoet within the target project's cwd.
        for project in projects:
            print(f"Running check tasks in {project}")
            tasks = extract_poe_tasks(project / "pyproject.toml")
            app = PoeThePoet(cwd=project)
            for sub_task in tasks_to_run:
                if sub_task in tasks:
                    print(f"Running task {sub_task} in {project}")
                    # run the subtask via the PoeThePoet app API
                    result = app(cli_args=[sub_task])
                    if result:
                        # Propagate the first non-zero exit code
                        sys.exit(result)
        sys.exit(0)

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
            print(f"Running task {task_name} in {project}")
            app = PoeThePoet(cwd=project)
            result = app(cli_args=poe_cli_args)
            if result:
                sys.exit(result)
        else:
            # This is not an error, some packages might not have all tasks
            pass


if __name__ == "__main__":
    main()