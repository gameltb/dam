import glob
import os
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
            result = app(cli_args=poe_cli_args)
            if result:
                print(f"Finished task {task_name} in {project} with args {task_args} with exit code {result}")
                sys.exit(result)
        else:
            # This is not an error, some packages might not have all tasks
            pass


if __name__ == "__main__":
    main()
