from __future__ import annotations

import asyncio
import csv
import datetime
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from dam.commands.analysis_commands import AnalysisCommand
from dam.core.transaction import WorldTransaction
from dam.functions import ecs_functions as dam_ecs_functions
from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component
from dam.system_events.base import BaseSystemEvent
from dam_archive.models import ArchiveMemberComponent
from dam_fs.models.file_location_component import FileLocationComponent
from rich.console import Console
from rich.traceback import Traceback
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from tqdm import tqdm
from typing_extensions import Annotated

from dam_app.state import get_world
from dam_app.utils.async_typer import AsyncTyper

app = AsyncTyper()


@dataclass
class VerifyArchiveContentsCommand(AnalysisCommand[None, BaseSystemEvent]):
    @classmethod
    def get_supported_types(cls) -> Dict[str, List[str]]:
        return {
            "mimetypes": [],
            "extensions": [".zip", ".rar", ".7z"],
        }


COMMAND_MAP = {
    "VerifyArchiveContentsCommand": VerifyArchiveContentsCommand,
}


@app.command(name="verify")
async def verify_assets(
    paths: Annotated[
        List[Path],
        typer.Argument(
            ...,
            help="Path to the asset file or directory of asset files.",
            exists=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    recursive: Annotated[
        bool,
        typer.Option("-r", "--recursive", help="Process directory recursively."),
    ] = False,
    process: Annotated[
        Optional[List[str]],
        typer.Option(
            "--process",
            "-p",
            help="Specify a command to run for a given MIME type or file extension, e.g., '.zip:VerifyArchiveContentsCommand'",
        ),
    ] = None,
    stop_on_error: Annotated[
        bool,
        typer.Option("--stop-on-error/--no-stop-on-error", help="Stop processing if an error occurs."),
    ] = True,
):
    """
    Verifies the integrity of local assets against the DAM.
    """
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    process_map: Dict[str, List[str]] = {}
    if process:
        for p in process:
            if ":" in p:
                try:
                    key, command_name = p.split(":", 1)
                    if key not in process_map:
                        process_map[key] = []
                    process_map[key].append(command_name)
                except ValueError:
                    typer.secho(
                        f"Invalid format for --process option: '{p}'. Must be 'key:CommandName'", fg=typer.colors.RED
                    )
                    raise typer.Exit(code=1)
            else:
                command_name = p
                command_class = COMMAND_MAP.get(command_name)
                if not command_class:
                    typer.secho(f"Unknown command '{command_name}' specified.", fg=typer.colors.RED)
                    raise typer.Exit(code=1)
                supported_types = command_class.get_supported_types()
                for mime_type in supported_types.get("mimetypes", []):
                    if mime_type not in process_map:
                        process_map[mime_type] = []
                    process_map[mime_type].append(command_name)
                for extension in supported_types.get("extensions", []):
                    if extension not in process_map:
                        process_map[extension] = []
                    process_map[extension].append(command_name)

    typer.echo("Starting asset verification process...")

    files_to_process: List[Path] = []
    for path in paths:
        if path.is_file():
            files_to_process.append(path)
        elif path.is_dir():
            pattern = "**/*" if recursive else "*"
            files_to_process.extend(p for p in path.glob(pattern) if p.is_file())

    if not files_to_process:
        typer.secho("No files found to process.", fg=typer.colors.YELLOW)
        return

    typer.echo(f"Found {len(files_to_process)} file(s) to process.")

    results: List[Dict[str, Any]] = []
    success_count = 0
    failed_count = 0
    skipped_count = 0
    error_count = 0

    async def _verify_file(session: AsyncSession, entity_id: int, file_path: Path, display_path: str):
        nonlocal success_count, failed_count, skipped_count, error_count
        try:
            stored_hash_orm = await dam_ecs_functions.get_component(session, entity_id, ContentHashSHA256Component)
            if not stored_hash_orm or not stored_hash_orm.hash_value:
                typer.secho(f"SKIP: No hash found in DAM for {display_path}", fg=typer.colors.YELLOW)
                results.append(
                    {
                        "file_path": display_path,
                        "calculated_hash": "N/A",
                        "dam_hash": "N/A",
                        "status": "SKIPPED (No Hash)",
                    }
                )
                skipped_count += 1
                return

            dam_hash = stored_hash_orm.hash_value.hex()
            proc = await asyncio.create_subprocess_shell(
                f"sha256sum '{file_path.as_posix()}'", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"Hash calculation failed for {display_path}: {stderr.decode()}")

            calculated_hash = stdout.decode().split()[0]
            if calculated_hash == dam_hash:
                status = "VERIFIED"
                success_count += 1
                typer.secho(f"OK: {display_path}", fg=typer.colors.GREEN)
            else:
                status = "FAILED"
                failed_count += 1
                typer.secho(f"FAIL: {display_path}", fg=typer.colors.RED)
            results.append(
                {"file_path": display_path, "calculated_hash": calculated_hash, "dam_hash": dam_hash, "status": status}
            )
        except Exception:
            error_count += 1
            console = Console()
            tqdm.write(f"Error verifying file {display_path}")
            console.print(Traceback())
            results.append(
                {"file_path": display_path, "calculated_hash": "ERROR", "dam_hash": "ERROR", "status": "ERROR"}
            )
            if stop_on_error:
                raise

    async def _verify_archive_contents(session: AsyncSession, archive_entity_id: int, archive_path: Path):
        nonlocal failed_count
        child_stmt = select(ArchiveMemberComponent).where(ArchiveMemberComponent.archive_entity_id == archive_entity_id)
        child_results = await session.execute(child_stmt)
        child_members = child_results.scalars().all()

        dam_members_map = {member.path_in_archive: member for member in child_members}
        dam_members_found: set[str] = set()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ext = archive_path.suffix.lower()
            cmd = ""
            if ext == ".zip":
                cmd = f"unzip -o '{archive_path.as_posix()}' -d '{tmp_path.as_posix()}'"
            elif ext == ".rar":
                cmd = f"unrar x -o+ '{archive_path.as_posix()}' '{tmp_path.as_posix()}/'"
            elif ext == ".7z":
                cmd = f"7z x -o'{tmp_path.as_posix()}' '{archive_path.as_posix()}'"
            else:
                tqdm.write(f"Unsupported archive type for verification: {ext}")
                return

            proc = await asyncio.create_subprocess_shell(
                cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                tqdm.write(f"Failed to extract archive {archive_path.name}: {stderr.decode()}")
                return

            extracted_files = {p.relative_to(tmp_path).as_posix(): p for p in tmp_path.glob("**/*") if p.is_file()}

            # Iterate over files in the archive
            for path_in_archive, file_path in extracted_files.items():
                display_path = f"{archive_path.name}/{path_in_archive}"
                member = dam_members_map.get(path_in_archive)

                if member:
                    await _verify_file(session, member.entity_id, file_path, display_path)
                    dam_members_found.add(path_in_archive)
                else:
                    # File exists in archive but not in DAM
                    status = "FAILED (Not in DAM)"
                    failed_count += 1
                    typer.secho(f"FAIL: {display_path} (Not in DAM)", fg=typer.colors.RED)
                    results.append(
                        {"file_path": display_path, "calculated_hash": "N/A", "dam_hash": "N/A", "status": status}
                    )

            # Check for files in DAM but not in archive
            for member in child_members:
                if member.path_in_archive not in dam_members_found:
                    display_path = f"{archive_path.name}/{member.path_in_archive}"
                    status = "FAILED (Not in Archive)"
                    failed_count += 1
                    typer.secho(f"FAIL: {display_path} (Not in Archive)", fg=typer.colors.RED)
                    results.append(
                        {"file_path": display_path, "calculated_hash": "N/A", "dam_hash": "N/A", "status": status}
                    )

    async def _process_entity(session: AsyncSession, entity_id: int, file_path: Path):
        await _verify_file(session, entity_id, file_path, file_path.name)
        ext = file_path.suffix.lower()
        commands_to_run = process_map.get(ext, [])
        for command_name in set(commands_to_run):
            if command_name == "VerifyArchiveContentsCommand":
                tqdm.write(f"Verifying contents of archive: {file_path.name}")
                await _verify_archive_contents(session, entity_id, file_path)

    with tqdm(total=len(files_to_process), desc="Verifying assets", unit="file") as pbar:
        async with target_world.get_context(WorldTransaction)() as tx:
            session = tx.session
            for file_path in files_to_process:
                pbar.set_postfix_str(file_path.name)
                try:
                    stmt = select(FileLocationComponent.entity_id).where(
                        FileLocationComponent.url == file_path.as_uri()
                    )
                    result = await session.execute(stmt)
                    entity_id = result.scalar_one_or_none()

                    if not entity_id:
                        typer.secho(f"SKIP: Asset not found in DAM for {file_path.name}", fg=typer.colors.YELLOW)
                        results.append(
                            {
                                "file_path": file_path.as_posix(),
                                "calculated_hash": "N/A",
                                "dam_hash": "N/A",
                                "status": "SKIPPED (Not Found)",
                            }
                        )
                        skipped_count += 1
                        continue

                    await _process_entity(session, entity_id, file_path)

                except Exception:
                    error_count += 1
                    console = Console()
                    tqdm.write(f"Error processing file {file_path.name}")
                    console.print(Traceback())
                    results.append(
                        {
                            "file_path": file_path.as_posix(),
                            "calculated_hash": "ERROR",
                            "dam_hash": "ERROR",
                            "status": "ERROR",
                        }
                    )
                    if stop_on_error:
                        raise
                finally:
                    pbar.update(1)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_filename = f"verification_report_{timestamp}.csv"
    with open(report_filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["file_path", "calculated_hash", "dam_hash", "status"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    typer.echo("\n--- Verification Summary ---")
    typer.secho(f"Report generated: {report_filename}", fg=typer.colors.CYAN)
    typer.echo(f"Successfully verified: {success_count}")
    typer.echo(f"Failed verification: {failed_count}")
    typer.echo(f"Skipped: {skipped_count}")
    typer.echo(f"Errors: {error_count}")
    typer.secho("Asset verification complete.", fg=typer.colors.GREEN)
