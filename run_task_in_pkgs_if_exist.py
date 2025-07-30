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


def main() -> None:
    pyproject_file = Path(__file__).parent / "pyproject.toml"
    projects = discover_projects(pyproject_file)

    if len(sys.argv) < 2:
        print("Please provide a task name")
        sys.exit(1)

    package_to_run = None
    # A simple arg parser to not add dependencies
    if "--package" in sys.argv:
        try:
            package_index = sys.argv.index("--package")
            package_to_run = sys.argv[package_index + 1]
            # Remove the --package and its value from sys.argv so it's not passed to poe
            sys.argv.pop(package_index)
            sys.argv.pop(package_index)
        except (ValueError, IndexError):
            print("Error: --package flag must be followed by a package name.")
            sys.exit(1)

    if package_to_run:
        projects = [p for p in projects if p.name == package_to_run]
        if not projects:
            print(f"Package '{package_to_run}' not found in workspace.")
            sys.exit(1)

    task_name = sys.argv[1]
    for project in projects:
        tasks = extract_poe_tasks(project / "pyproject.toml")
        if task_name in tasks:
            print(f"Running task {task_name} in {project}")
            app = PoeThePoet(cwd=project)
            result = app(cli_args=sys.argv[1:])
            if result:
                sys.exit(result)
        else:
            # This is not an error, some packages might not have all tasks
            pass


if __name__ == "__main__":
    main()
