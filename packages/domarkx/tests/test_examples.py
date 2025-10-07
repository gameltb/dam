import os

import pytest

from domarkx.utils.chat_doc_parser import MarkdownLLMParser


def get_example_files() -> list[str]:
    example_dir = os.path.join(os.path.dirname(__file__), "..", "examples")
    if not os.path.isdir(example_dir):
        return []
    return [os.path.join(example_dir, f) for f in os.listdir(example_dir) if f.endswith(".md")]


@pytest.mark.parametrize("filepath", get_example_files())
def test_example_conformance(filepath: str) -> None:
    """Tests that example markdown files conform to the domarkx documentation format."""
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()

        parser = MarkdownLLMParser()
        doc = parser.parse(content)

        # 1. Check for optional YAML front matter
        assert isinstance(doc.global_metadata, dict), "Front matter must be a dictionary if present."

        # 2. Check for optional session-config block
        if doc.session_config is not None:
            assert isinstance(doc.session_config, dict), "Session config must be a dictionary if present."

        # 3. Check conversation format (h2 headings)
        for i, message in enumerate(doc.conversation):
            assert hasattr(message, "speaker"), f"Message {i} in {filepath} is missing 'speaker'."
            assert isinstance(message.speaker, str), f"Speaker in message {i} in {filepath} must be a string."

            # 4. Check message structure
            # This is implicitly checked by the parser, if it doesn't raise an error, the structure is valid.
            # We can add more specific checks here if needed, for example, checking the content of metadata.
            if message.metadata:
                assert isinstance(message.metadata, dict), (
                    f"Metadata in message {i} in {filepath} must be a dictionary."
                )

    except Exception as e:
        pytest.fail(f"Failed to parse or validate {filepath}: {e}")
