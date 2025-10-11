"""Tests for the `report` CLI command."""

import csv
import zipfile
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from dam_app.main import app

runner = CliRunner()


def test_report_duplicates_command(tmp_path: Path, postgresql: Any):
    """Test the `report duplicates` command."""
    # Create a test world
    db_url = (
        f"postgresql://{postgresql.user}:{postgresql.password}@{postgresql.host}:{postgresql.port}/{postgresql.dbname}"
    )
    config_path = tmp_path / "dam.toml"
    with config_path.open("w") as f:
        f.write(
            f"""
[worlds.test_world]
database = "{db_url}"
plugins = ["dam.core", "dam.fs", "dam.archive", "dam.psp"]
"""
        )

    # Create a file
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello")

    # Create an archive with the same file
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("test.txt", "hello")

    # Add the file to the world
    runner.invoke(
        app,
        [
            "--config",
            str(config_path),
            "--world",
            "test_world",
            "assets",
            "add",
            str(file_path),
        ],
        catch_exceptions=False,
    )

    # Add the archive to the world
    runner.invoke(
        app,
        [
            "--config",
            str(config_path),
            "--world",
            "test_world",
            "assets",
            "add",
            str(zip_path),
            "--process",
            ".zip:IngestArchive",
        ],
        catch_exceptions=False,
    )

    # Run the report duplicates command
    result = runner.invoke(
        app,
        [
            "--config",
            str(config_path),
            "--world",
            "test_world",
            "report",
            "duplicates",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "Duplicate Files Report" in result.stdout
    assert "Total wasted space: 5 bytes" in result.stdout
    assert "Filesystem" in result.stdout
    assert "Archive" in result.stdout

    # Test CSV output
    csv_path = tmp_path / "duplicates.csv"
    result = runner.invoke(
        app,
        [
            "--config",
            str(config_path),
            "--world",
            "test_world",
            "report",
            "duplicates",
            "--csv",
            str(csv_path),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert csv_path.exists()

    with csv_path.open("r") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == [
            "Entity ID",
            "SHA256",
            "Size (bytes)",
            "Locations",
            "Wasted Space (bytes)",
            "Paths",
        ]
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0][2] == "5"  # Size
        assert rows[0][3] == "2"  # Locations
        assert rows[0][4] == "5"  # Wasted space
