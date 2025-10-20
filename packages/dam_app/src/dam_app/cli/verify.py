"""Defines the CLI for verifying the integrity of local assets against the DAM."""

from __future__ import annotations

import asyncio
import csv
import datetime
import hashlib
import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import aiofiles
import typer
from dam.commands.analysis_commands import AnalysisCommand
from dam.core.transaction import WorldTransaction
from dam.core.world import World
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

from dam_app.state import get_world
from dam_app.utils.async_typer import AsyncTyper

app = AsyncTyper()


@dataclass
class VerifyArchiveContentsCommand(AnalysisCommand[None, BaseSystemEvent]):
    """A command to trigger verification of archive contents."""

    @classmethod
    def get_supported_types(cls) -> dict[str, list[str]]:
        """Return the supported file types for this command."""
        return {
            "mimetypes": [],
            "extensions": [".zip", ".rar", ".7z"],
        }


COMMAND_MAP = {
    "VerifyArchiveContentsCommand": VerifyArchiveContentsCommand,
}


async def _get_sha256(file_path: Path) -> str:
    """Calculate the SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    async with aiofiles.open(file_path, "rb") as f:
        while True:
            data = await f.read(65536)  # Read in 64k chunks
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


async def _verify_file(session: AsyncSession, entity_id: int, file_path: Path, display_path: str) -> dict[str, Any]:
    """Verify a single file's hash against the one stored in the DAM."""
    try:
        stored_hash_orm = await dam_ecs_functions.get_component(session, entity_id, ContentHashSHA256Component)
        if not stored_hash_orm or not stored_hash_orm.hash_value:
            typer.secho(f"SKIP: No hash found in DAM for {display_path}", fg=typer.colors.YELLOW)
            return {"file_path": display_path, "status": "SKIPPED (No Hash)"}

        dam_hash = stored_hash_orm.hash_value.hex()
        calculated_hash = await _get_sha256(file_path)

        if calculated_hash == dam_hash:
            status = "VERIFIED"
            typer.secho(f"OK: {display_path}", fg=typer.colors.GREEN)
        else:
            status = "FAILED"
            typer.secho(f"FAIL: {display_path}", fg=typer.colors.RED)
        return {"file_path": display_path, "calculated_hash": calculated_hash, "dam_hash": dam_hash, "status": status}
    except Exception:
        console = Console()
        tqdm.write(f"Error verifying file {display_path}")
        console.print(Traceback())
        return {"file_path": display_path, "calculated_hash": "ERROR", "dam_hash": "ERROR", "status": "ERROR"}


async def _extract_archive(archive_path: Path, tmp_path: Path) -> bool:
    """Extract an archive to a temporary directory."""
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
        return False

    proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        tqdm.write(f"Failed to extract archive {archive_path.name}: {stderr.decode()}")
        return False
    return True


async def _verify_archive_contents(
    session: AsyncSession, archive_entity_id: int, archive_path: Path
) -> list[dict[str, Any]]:
    """Verify the contents of an archive against DAM records."""
    results: list[dict[str, Any]] = []
    child_stmt = select(ArchiveMemberComponent).where(ArchiveMemberComponent.archive_entity_id == archive_entity_id)
    child_results = await session.execute(child_stmt)
    child_members = child_results.scalars().all()
    dam_members_map = {member.path_in_archive: member for member in child_members}
    dam_members_found: set[str] = set()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        if not await _extract_archive(archive_path, tmp_path):
            return []

        extracted_files = {p.relative_to(tmp_path).as_posix(): p for p in tmp_path.glob("**/*") if p.is_file()}

        for path_in_archive, file_path in extracted_files.items():
            display_path = f"{archive_path.name}/{path_in_archive}"
            member = dam_members_map.get(path_in_archive)
            if member:
                result = await _verify_file(session, member.entity_id, file_path, display_path)
                results.append(result)
                dam_members_found.add(path_in_archive)
            else:
                typer.secho(f"FAIL: {display_path} (Not in DAM)", fg=typer.colors.RED)
                results.append({"file_path": display_path, "status": "FAILED (Not in DAM)"})

        for member in child_members:
            if member.path_in_archive not in dam_members_found:
                display_path = f"{archive_path.name}/{member.path_in_archive}"
                typer.secho(f"FAIL: {display_path} (Not in Archive)", fg=typer.colors.RED)
                results.append({"file_path": display_path, "status": "FAILED (Not in Archive)"})
    return results


async def _process_entity(
    session: AsyncSession, entity_id: int, file_path: Path, process_map: dict[str, list[str]]
) -> list[dict[str, Any]]:
    """Process a single entity for verification."""
    results: list[dict[str, Any]] = []
    file_result = await _verify_file(session, entity_id, file_path, file_path.name)
    results.append(file_result)

    ext = file_path.suffix.lower()
    commands_to_run = process_map.get(ext, [])
    for command_name in set(commands_to_run):
        if command_name == "VerifyArchiveContentsCommand":
            tqdm.write(f"Verifying contents of archive: {file_path.name}")
            archive_results = await _verify_archive_contents(session, entity_id, file_path)
            results.extend(archive_results)
    return results


def _update_counts(results: list[dict[str, Any]]) -> tuple[int, int, int]:
    """Update success, failed, and skipped counts based on verification results."""
    success = sum(1 for r in results if r["status"] == "VERIFIED")
    failed = sum(1 for r in results if r["status"].startswith("FAILED"))
    skipped = sum(1 for r in results if r["status"].startswith("SKIPPED"))
    return success, failed, skipped


async def verify_assets_logic(
    world: World,
    paths: list[Path],
    recursive: bool,
    process_map: dict[str, list[str]],
    stop_on_error: bool,
    pbar: tqdm[Any] | None = None,
) -> tuple[list[dict[str, Any]], int, int, int, int]:
    """Core logic for verifying assets."""
    files_to_process: list[Path] = []
    for path in paths:
        if path.is_file():
            files_to_process.append(path)
        elif path.is_dir():
            pattern = "**/*" if recursive else "*"
            files_to_process.extend(p for p in path.glob(pattern) if p.is_file())

    all_results: list[dict[str, Any]] = []
    success_count, failed_count, skipped_count, error_count = 0, 0, 0, 0

    if not files_to_process:
        return all_results, success_count, failed_count, skipped_count, error_count

    if pbar:
        pbar.total = len(files_to_process)

    async with world.get_context(WorldTransaction)() as tx:
        session = tx.session
        for file_path in files_to_process:
            if pbar:
                pbar.set_postfix_str(file_path.name)
            try:
                stmt = select(FileLocationComponent.entity_id).where(FileLocationComponent.url == file_path.as_uri())
                result = await session.execute(stmt)
                entity_id = result.scalar_one_or_none()

                if not entity_id:
                    typer.secho(f"SKIP: Asset not found in DAM for {file_path.name}", fg=typer.colors.YELLOW)
                    all_results.append({"file_path": file_path.as_posix(), "status": "SKIPPED (Not Found)"})
                    skipped_count += 1
                    continue

                entity_results = await _process_entity(session, entity_id, file_path, process_map)
                all_results.extend(entity_results)
                s, f, sk = _update_counts(entity_results)
                success_count += s
                failed_count += f
                skipped_count += sk
            except Exception:
                error_count += 1
                console = Console()
                tqdm.write(f"Error processing file {file_path.name}")
                console.print(Traceback())
                all_results.append({"file_path": file_path.as_posix(), "status": "ERROR"})
                if stop_on_error:
                    raise
            finally:
                if pbar:
                    pbar.update(1)

    return all_results, success_count, failed_count, skipped_count, error_count


@app.command(name="verify")
async def verify_assets(
    paths: Annotated[
        list[Path],
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
        list[str] | None,
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
    """Verify the integrity of local assets against the DAM."""
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    process_map: dict[str, list[str]] = {}
    if process:
        for p in process:
            key, command_name = p.split(":", 1)
            if key not in process_map:
                process_map[key] = []
            process_map[key].append(command_name)

    typer.echo("Starting asset verification process...")

    files_to_process: list[Path] = []
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

    with tqdm(total=len(files_to_process), desc="Verifying assets", unit="file") as pbar:
        all_results, success_count, failed_count, skipped_count, error_count = await verify_assets_logic(
            world=target_world,
            paths=paths,
            recursive=recursive,
            process_map=process_map,
            stop_on_error=stop_on_error,
            pbar=pbar,
        )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_filename = f"verification_report_{timestamp}.csv"

    string_buffer = io.StringIO()
    fieldnames = ["file_path", "calculated_hash", "dam_hash", "status"]
    writer = csv.DictWriter(string_buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_results)

    async with aiofiles.open(report_filename, "w", newline="", encoding="utf-8") as csvfile:
        await csvfile.write(string_buffer.getvalue())

    typer.echo("\n--- Verification Summary ---")
    typer.secho(f"Report generated: {report_filename}", fg=typer.colors.CYAN)
    typer.echo(f"Successfully verified: {success_count}")
    typer.echo(f"Failed verification: {failed_count}")
    typer.echo(f"Skipped: {skipped_count}")
    typer.echo(f"Errors: {error_count}")
    typer.secho("Asset verification complete.", fg=typer.colors.GREEN)
