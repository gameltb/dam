import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mistune

try:
    import frontmatter
except ImportError:
    frontmatter = None


@dataclass
class CodeBlock:
    language: Optional[str] = None
    attrs: Optional[str] = None
    code: str = ""


@dataclass
class SessionMetadata:
    session_config: dict = field(default_factory=lambda: {})
    session_setup_code: Optional[CodeBlock] = None


@dataclass
class Message:
    speaker: str
    content: str
    metadata: dict = field(default_factory=lambda: {})


@dataclass
class ParsedDocument:
    global_metadata: dict = field(default_factory=lambda: {})
    config: SessionMetadata = field(default_factory=SessionMetadata)
    conversation: List[Message] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    raw_lines: List[str] = field(default_factory=list, repr=False)


def find_file_blocks(text):
    """
    使用正则表达式查找并解析特殊格式的文件代码块。
    文件名是可选的。

    Args:
        text (str): 包含文件代码块的字符串。

    Returns:
        list: 一个包含解析出的块信息的字典列表。
    """
    # 更新后的正则表达式，文件名部分是可选的
    # (````|```)              # 第1组: 捕获4个或3个反引号
    # (\w*)                    # 第2组: 捕获可选的语言标识符
    # (?:\s*name=([\w.-]+))?   # 可选的非捕获组，内部包含
    #                          # 第3组: 用于文件名的捕获组
    # \n                       # 匹配换行符
    # (.*?)                    # 第4组: 非贪婪地捕获所有内容（文件内容）
    # \n                       # 匹配换行符
    # \1                       # 匹配与第1组完全相同的内容 (结束的反引号)
    pattern = r"(````|```)(\w*)(?:\s*name=([\S]+))?\n(.*)\n\1"

    # 使用 re.finditer 和 re.DOTALL 标志
    matches = re.finditer(pattern, text, re.DOTALL)

    results = []
    for i, match in enumerate(matches):
        # 组2是语言, 组3是文件名, 组4是内容
        # 如果文件名不存在 (组3为None)，则提供一个默认值
        block_info = {
            "id": i + 1,
            "language": match.group(2) or None,  # 使用 'or' 更简洁
            "filename": match.group(3) or None,  # 如果组3是None，则返回 'N/A'
            "content": match.group(4),
        }
        results.append(block_info)

    return results


class BlockParser(mistune.BlockParser):
    def parse_block_quote(self, m, state) -> int:
        text, end_pos = self.extract_block_quote(m, state)
        token = {"type": "block_quote", "raw": text}
        if end_pos:
            state.prepend_token(token)
            return end_pos
        state.append_token(token)
        return state.cursor


class MarkdownLLMParser:
    CONFIG_BLOCK_LANG = "session-config"
    MESSAGE_METADATA_LANG = "msg-metadata"
    CODE_BLOCK_REGEX = re.compile(r"```(?:\s*([\w\+\-]+))?(?:\s*([\S]+))?\n([\s\S]*?)```", re.MULTILINE)
    INCLUSION_REGEX = re.compile(
        "^(?P<indent>[ \\t]*)(?P<quote_prefix>>[ \\t]*)?\\[include\\]\\((?P<path>[^)]+)\\)[ \\t]*$",
        re.IGNORECASE | re.MULTILINE,
    )

    def __init__(self):
        block_parser = BlockParser()
        self.markdown_parser = mistune.create_markdown(renderer="ast")
        self.markdown_parser.block = block_parser
        self.document = ParsedDocument()

    def _get_paragraph_text_from_list_item(self, paragraph_node: Dict[str, Any]) -> str:
        return "".join((c.get("raw", "") for c in paragraph_node.get("children", [])))

    def _get_node_text_content(self, node: Dict[str, Any]) -> str:
        if "children" in node:
            return "".join((child.get("raw", "") for child in node["children"] if child.get("type") == "text"))
        return node.get("raw", "")

    def _fetch_inclusion_content(self, target_path: str, base_dir: str) -> str:
        full_path = os.path.normpath(os.path.join(base_dir, target_path))
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Included file not found: {full_path}")
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()

    def _format_as_blockquote(self, text: str) -> str:
        """Prefixes each line of the text with '> ' to format it as a blockquote."""
        if not text:
            return ""
        lines = text.splitlines()
        return "\n".join((f"> {line}" for line in lines))

    def _resolve_inclusions(
        self,
        md_content: str,
        content_abs_path: str,
        content_base_dir: str,
        processing_stack: set,
    ) -> str:
        """
        Recursively resolves [include] directives.
        Args:
            md_content: The markdown content to process.
            content_abs_path: Absolute path of the md_content (can be conceptual for strings).
            content_base_dir: Absolute base directory for resolving relative paths in md_content.
            processing_stack: Set of absolute paths currently in the recursion stack for cycle detection.
        """
        if content_abs_path in processing_stack:
            err_msg = f"Circular inclusion detected: {content_abs_path} is already being processed."
            self.document.errors.append(err_msg)
            return f"> **ERROR**: {err_msg}"
        processing_stack.add(content_abs_path)
        output_content = md_content
        while True:
            match = self.INCLUSION_REGEX.search(output_content)
            if not match:
                break
            indent_prefix = match.group("indent") or ""
            is_in_blockquote_directive = bool(match.group("quote_prefix"))
            path_spec = match.group("path")
            parts = path_spec.split("#", 1)
            target_file_rel_path = parts[0]
            msg_index: Optional[int] = None
            if len(parts) > 1:
                msg_index_str = parts[1]
                try:
                    msg_index = int(msg_index_str)
                except ValueError:
                    err_msg = f"Invalid message index '{msg_index_str}' in include directive for '{target_file_rel_path}'. Must be an integer."
                    self.document.errors.append(err_msg)
                    replacement_content = f"> **ERROR**: {err_msg}"
                    output_content = (
                        output_content[: match.start()]
                        + indent_prefix
                        + replacement_content
                        + output_content[match.end() :]
                    )
                    continue
            replacement_content = ""
            try:
                included_file_abs_path = os.path.normpath(os.path.join(content_base_dir, target_file_rel_path))
                included_file_base_dir = os.path.dirname(included_file_abs_path)
                included_md_raw = self._fetch_inclusion_content(target_file_rel_path, content_base_dir)
                if msg_index is not None:
                    sub_parser = MarkdownLLMParser()
                    parsed_included_doc = sub_parser.parse(
                        included_md_raw,
                        source_path=included_file_abs_path,
                        _processing_stack=processing_stack,
                    )
                    self.document.errors.extend(sub_parser.document.errors)
                    messages = parsed_included_doc.conversation
                    if not messages:
                        err_msg = f"No messages found in '{target_file_rel_path}' to select index {msg_index}."
                        self.document.errors.append(err_msg)
                        replacement_content = f"> **ERROR**: {err_msg}"
                    else:
                        num_messages = len(messages)
                        effective_msg_index = msg_index
                        if msg_index < 0:
                            effective_msg_index = num_messages + msg_index
                        if 0 <= effective_msg_index < num_messages:
                            replacement_content = messages[effective_msg_index].content
                        else:
                            err_msg = f"Message index {msg_index} (resolved to {effective_msg_index}) out of bounds for {target_file_rel_path} ({num_messages} messages)."
                            self.document.errors.append(err_msg)
                            replacement_content = f"> **ERROR**: {err_msg}"
                else:
                    replacement_content = self._resolve_inclusions(
                        included_md_raw,
                        included_file_abs_path,
                        included_file_base_dir,
                        processing_stack,
                    )
            except FileNotFoundError as e:
                self.document.errors.append(str(e))
                replacement_content = f"> **ERROR**: Included file not found: {target_file_rel_path}"
            except Exception as e:
                err_detail = f"Error processing inclusion for '{target_file_rel_path}': {type(e).__name__} - {e}"
                self.document.errors.append(err_detail)
                replacement_content = f"> **ERROR**: Could not process inclusion for `{target_file_rel_path}`"
            if is_in_blockquote_directive:
                replacement_content = self._format_as_blockquote(replacement_content)
            final_replacement_lines = [f"{indent_prefix}{line}" for line in replacement_content.splitlines()]
            if not final_replacement_lines and replacement_content:
                final_replacement = indent_prefix + replacement_content
            elif not final_replacement_lines and (not replacement_content):
                final_replacement = ""
            else:
                final_replacement = "\n".join(final_replacement_lines)
                if (
                    replacement_content.endswith("\n")
                    and (not final_replacement.endswith("\n"))
                    and (replacement_content != "\n")
                ):
                    pass
            output_content = output_content[: match.start()] + final_replacement + output_content[match.end() :]
        processing_stack.remove(content_abs_path)
        return output_content

    def parse(
        self,
        md_content: str,
        source_path: str = ".",
        _processing_stack: Optional[set] = None,
        resolve_inclusions=True,
    ) -> ParsedDocument:
        self.document = ParsedDocument()
        current_processing_stack = _processing_stack if _processing_stack is not None else set()
        content_abs_path: str
        content_base_dir: str
        normalized_source_path = os.path.abspath(source_path)
        if os.path.isfile(normalized_source_path):
            content_abs_path = normalized_source_path
            content_base_dir = os.path.dirname(content_abs_path)
        else:
            content_base_dir = normalized_source_path
            content_abs_path = os.path.join(content_base_dir, f"__InMemoryContent_{id(md_content)}__")
            if not os.path.exists(content_base_dir) and content_base_dir == os.path.abspath("."):
                pass
            elif not os.path.exists(content_base_dir):
                self.document.errors.append(
                    f"Base directory for inclusions '{content_base_dir}' from source_path '{source_path}' does not exist."
                )
                content_base_dir = os.path.abspath(".")
        if resolve_inclusions:
            try:
                markdown_body = self._resolve_inclusions(
                    md_content,
                    content_abs_path,
                    content_base_dir,
                    current_processing_stack,
                )
            except Exception as e:
                self.document.errors.append(f"Fatal error during inclusion resolution: {e}")
                markdown_body = md_content
        else:
            markdown_body = md_content
        self.document.raw_lines = markdown_body.splitlines(keepends=True)
        if frontmatter:
            try:
                fm_post = frontmatter.loads(markdown_body)
                self.document.global_metadata = fm_post.metadata
                markdown_body = fm_post.content
            except Exception as e:
                self.document.errors.append(f"Error parsing YAML front matter: {e}")
        elif md_content.startswith("---"):
            self.document.errors.append(
                "YAML front matter detected, but 'python-frontmatter' library is not installed."
            )
        ast_nodes = self.markdown_parser(markdown_body)
        if not isinstance(ast_nodes, list):
            self.document.errors.append(f"Markdown parsing did not return a list of nodes. Got: {type(ast_nodes)}")
            ast_nodes = []
        i = 0
        config_parsed = False
        while i < len(ast_nodes):
            node = ast_nodes[i]
            node_type = node.get("type")
            if (
                not config_parsed
                and node_type == "block_code"
                and (self.CONFIG_BLOCK_LANG in node.get("attrs", {}).get("info", ""))
            ):
                try:
                    config_data = json.loads(node.get("raw", "{}"))
                    parsed_post_config_block = None
                    temp_i = i + 1
                    while temp_i < len(ast_nodes) and ast_nodes[temp_i].get("type") == "blank_line":
                        temp_i += 1
                    if temp_i < len(ast_nodes) and ast_nodes[temp_i].get("type") == "block_code":
                        post_config_node = ast_nodes[temp_i]
                        parsed_post_config_block = CodeBlock(
                            language=post_config_node.get("attrs", {}).get("info"),
                            code=post_config_node.get("raw", "").strip("\n"),
                        )
                        i = temp_i
                    self.document.config = SessionMetadata(
                        session_config=config_data,
                        session_setup_code=parsed_post_config_block,
                    )
                except json.JSONDecodeError as e:
                    self.document.errors.append(f"Error parsing session-config JSON: {e}")
                config_parsed = True
                i += 1
                continue
            if node_type == "heading" and node.get("style", "") == "atx" and (node.get("attrs", {}).get("level") == 2):
                config_parsed = True
                speaker_text = self._get_node_text_content(node).strip()
                current_message_obj = Message(speaker=speaker_text, content="")
                msg_meta_node_idx = i + 1
                while msg_meta_node_idx < len(ast_nodes) and ast_nodes[msg_meta_node_idx].get("type") == "blank_line":
                    msg_meta_node_idx += 1
                if (
                    msg_meta_node_idx < len(ast_nodes)
                    and ast_nodes[msg_meta_node_idx].get("type") == "block_code"
                    and (self.MESSAGE_METADATA_LANG in ast_nodes[msg_meta_node_idx].get("attrs", {}).get("info", ""))
                ):
                    meta_code_block_node = ast_nodes[msg_meta_node_idx]
                    try:
                        meta_data = json.loads(meta_code_block_node.get("raw", "{}"))
                        current_message_obj.metadata = meta_data
                    except json.JSONDecodeError as e:
                        self.document.errors.append(f"Error parsing msg-metadata JSON for '{speaker_text}': {e}")
                    i = msg_meta_node_idx
                msg_content_node_idx = i + 1
                while (
                    msg_content_node_idx < len(ast_nodes)
                    and ast_nodes[msg_content_node_idx].get("type") == "blank_line"
                ):
                    msg_content_node_idx += 1
                if (
                    msg_content_node_idx < len(ast_nodes)
                    and ast_nodes[msg_content_node_idx].get("type") == "block_quote"
                ):
                    message_blockquote_node = ast_nodes[msg_content_node_idx]
                    current_message_obj.content = message_blockquote_node.get("raw", "")
                    i = msg_content_node_idx
                else:
                    err_node_type = (
                        ast_nodes[msg_content_node_idx].get("type")
                        if msg_content_node_idx < len(ast_nodes)
                        else "nothing"
                    )
                    self.document.errors.append(
                        f"Expected block_quote for message content for speaker '{speaker_text}', found {err_node_type} at line ~{(node.get('line', '?') if node else '?')}. Content may be missing."
                    )
                self.document.conversation.append(current_message_obj)
                i += 1
                continue
            i += 1
        return self.document

    def get_message_and_code_block(
        self, messageIndex: int, codeBlockInMessageIndex: Optional[int] = None
    ) -> Tuple[Optional[Message], Optional[CodeBlock]]:
        """
        Retrieves a specific message and a specific code block within that message's content.

        Args:
            messageIndex: The index of the message in the conversation list.
            codeBlockInMessageIndex: The index of the code block within the found message's content.

        Returns:
            A tuple containing (Message, CodeBlock).
            Returns (None, None) if the messageIndex is out of bounds.
            Returns (Message, None) if messageIndex is valid but codeBlockInMessageIndex is out of bounds.
        """
        if not 0 <= messageIndex < len(self.document.conversation):
            return (None, None)
        message_obj = self.document.conversation[messageIndex]
        if codeBlockInMessageIndex is None:
            return (message_obj, None)
        actual_message_content = message_obj.content
        found_code_blocks: List[CodeBlock] = []
        for match in find_file_blocks(actual_message_content):
            found_code_blocks.append(
                CodeBlock(language=match["language"], attrs=match["filename"], code=match["content"])
            )
        if not 0 <= codeBlockInMessageIndex < len(found_code_blocks):
            return (message_obj, None)
        return (message_obj, found_code_blocks[codeBlockInMessageIndex])


def append_message(writer: io.StringIO, message: Message):
    writer.write(
        f"""
## {message.speaker}

```json msg-metadata
{json.dumps(message.metadata, indent=2, ensure_ascii=False)}
```

{"\n".join(("> " + line for line in message.content.splitlines()))}
"""
    )
