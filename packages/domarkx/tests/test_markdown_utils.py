from domarkx.utils.markdown_utils import find_code_blocks


def test_find_code_blocks() -> None:
    text = """
Some text
```python name=test.py
print("hello")
```
Some more text
"""
    blocks = find_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].language == "python"
    assert blocks[0].attrs == "test.py"
    assert blocks[0].code == 'print("hello")\n'


def test_find_macros() -> None:
    pass
