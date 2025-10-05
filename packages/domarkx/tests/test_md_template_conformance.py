import glob
from typing import List

from typer.testing import CliRunner

from domarkx.cli import cli_app
from domarkx.utils.chat_doc_parser import MarkdownLLMParser

runner = CliRunner()


def test_all_templates_md_conformance() -> None:
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

        all_errors: List[str] = []
        for md_path in md_files:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
                try:
                    parsed = parser.parse(content)
                    if not isinstance(parsed.global_metadata, dict):
                        all_errors.append(f"{md_path}: global_metadata is not a dict")
                    elif "title" not in parsed.global_metadata:
                        all_errors.append(f"{md_path}: Missing 'title' in global_metadata")
                    if not isinstance(parsed.session_config, dict):
                        all_errors.append(f"{md_path}: session_config is not a dict")
                    if len(parsed.conversation) == 0:
                        all_errors.append(f"{md_path}: No conversation found")
                    for msg in parsed.conversation:
                        if not msg.speaker:
                            all_errors.append(f"{md_path}: Message missing speaker")
                        # Check that each message contains at least a code block or content
                        has_code = len(msg.code_blocks) > 0
                        has_content = msg.content is not None and msg.content.strip() != ""
                        if not has_code and not has_content:
                            all_errors.append(f"{md_path}: Message by {msg.speaker} has no code or content")
                except ValueError as e:
                    all_errors.append(f"Error parsing {md_path}: {e}")
        assert not all_errors, "Markdown template errors found:\n" + "\n".join(all_errors)
