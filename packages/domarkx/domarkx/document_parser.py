import pathlib
from domarkx.utils.chat_doc_parser import MarkdownLLMParser

def parse_document(doc_path: pathlib.Path):
    with doc_path.open() as f:
        md_content = f.read()

    parser = MarkdownLLMParser()
    return parser.parse(md_content, source_path=str(doc_path.absolute()))
