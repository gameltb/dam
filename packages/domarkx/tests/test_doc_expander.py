import pathlib

import pytest

from domarkx.macro_expander import DocExpander


@pytest.fixture
def base_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path


@pytest.fixture
def doc_expander(base_dir: pathlib.Path) -> DocExpander:
    return DocExpander(base_dir=str(base_dir))


def test_expand_macros_in_messages(doc_expander: DocExpander, base_dir: pathlib.Path) -> None:
    # Create a file to be included
    include_file = base_dir / "include.md"
    include_file.write_text("included content")

    md = """## User

> This is the first message. [@include](domarkx://include?path=include.md)

## Assistant

> This is the second message.
"""
    expanded_doc = doc_expander.expand(md)
    assert len(expanded_doc.conversation) == 2
    assert expanded_doc.conversation[0].content is not None
    assert expanded_doc.conversation[0].content.strip() == "This is the first message. included content"
    assert expanded_doc.conversation[1].content is not None
    assert expanded_doc.conversation[1].content.strip() == "This is the second message."
