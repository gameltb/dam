import glob
from domarkx.utils.chat_doc_parser import MarkdownLLMParser
import os
import shutil
import subprocess
from typer.testing import CliRunner
from domarkx.cli import cli_app

runner = CliRunner()


def test_all_templates_md_conformance():
    """
    Initialize a project and check every .md file in the project for MarkdownLLMParser conformance.
    """
    with runner.isolated_filesystem():
        result = runner.invoke(cli_app, ["init"])
        assert result.exit_code == 0
        # Search for all .md files in the project, not just templates
        md_files = glob.glob("**/*.md", recursive=True)
        assert md_files, "No .md files found in project"
        parser = MarkdownLLMParser()
        from typing import List

        all_errors: List[str] = []
        for md_path in md_files:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            parsed = parser.parse(content, source_path=md_path)
            if not isinstance(parsed.global_metadata, dict):
                all_errors.append(f"{md_path}: global_metadata is not a dict")
            elif "title" not in parsed.global_metadata:
                all_errors.append(f"{md_path}: Missing 'title' in global_metadata")
            if not isinstance(parsed.config.session_config, dict):
                all_errors.append(f"{md_path}: session_config is not a dict")
            if len(parsed.conversation) == 0:
                all_errors.append(f"{md_path}: No conversation found")
            for msg in parsed.conversation:
                if not msg.speaker:
                    all_errors.append(f"{md_path}: Message missing speaker")
                # Check that each message contains at least a code block or blockquote
                has_code = False
                has_blockquote = False
                lines = msg.content.splitlines()
                in_code = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_code = not in_code
                        if in_code:
                            has_code = True
                        continue
                    if in_code:
                        continue
                    if line.strip().startswith(">"):
                        has_blockquote = True
                if not (has_code or has_blockquote):
                    all_errors.append(
                        f"{md_path}: Message must contain at least one code block or blockquote. Speaker: {msg.speaker}"
                    )
        assert not all_errors, "Markdown template errors found:\n" + "\n".join(all_errors)
