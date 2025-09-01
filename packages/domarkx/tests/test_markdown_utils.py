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
    text = """
This is a test with a macro: [@my_macro](domarkx://run?arg1=val1&arg2=val2).
This is a normal link: [google](https://google.com).
This is an escaped macro: [@@not_a_macro](domarkx://run).
"""
    pass
