import ast
import pathlib
from domarkx.utils.chat_doc_parser import MarkdownLLMParser, ParsedDocument
import os
from domarkx.utils.markdown_utils import find_macros, find_first_macro
from domarkx.agents.resume_funcall_assistant_agent import ResumeFunCallAssistantAgent


import os
from domarkx.utils.markdown_utils import find_macros, find_first_macro


class MacroExpander:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.macros = {
            "include": self._include_macro,
        }

    def expand(self, content: str, parameters: dict = None) -> str:
        """Expands macros in the content sequentially: find and expand the first macro, repeat until all macros are processed."""
        if parameters is None:
            parameters = {}

        expanded_content = content
        while True:
            match = find_first_macro(expanded_content)
            if not match:
                break
            macro_name = match.group(2)
            macro_text = match.group(1)
            macro_value = parameters.get(macro_name, parameters.get(macro_text, match.group(0)))
            # If macro_name is a special handler (e.g. include), use handler
            if hasattr(self, f"_{macro_name}_macro"):
                macro_obj = type("Macro", (), {})()
                macro_obj.command = macro_name
                macro_obj.link_text = macro_text
                macro_obj.url = f"domarkx://{macro_name}"
                macro_obj.params = {}
                macro_value = getattr(self, f"_{macro_name}_macro")(macro_obj, expanded_content)
            expanded_content = expanded_content[: match.start()] + str(macro_value) + expanded_content[match.end() :]
        return expanded_content

    def _include_macro(self, macro, content):
        """Handles the @include macro."""
        path = macro.params.get("path")
        if not path:
            return content

        include_path = pathlib.Path(path)
        if not include_path.is_absolute():
            include_path = pathlib.Path(self.base_dir) / include_path

        if include_path.exists():
            include_content = include_path.read_text()
            return content.replace(f"[@{macro.link_text}]({macro.url})", include_content, 1)
        else:
            return content


class Session:
    def __init__(self, doc_path: pathlib.Path):
        self.doc_path = doc_path
        self.doc = self._parse_document()
        self.agent = None
        self.tool_executors = []

    def _parse_document(self):
        with self.doc_path.open() as f:
            md_content = f.read()

        parser = MarkdownLLMParser()
        return parser.parse(md_content, source_path=str(self.doc_path.absolute()))

    def _get_last_expression(self, code):
        try:
            tree = ast.parse(code)
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = tree.body.pop()
                return ast.unparse(last_expr), ast.unparse(tree)
            return None, code
        except SyntaxError:
            return None, code

    async def setup(self):
        setup_script_blocks = self.doc.get_code_blocks(language="python", attrs="setup-script")
        if not setup_script_blocks:
            raise ValueError("No setup-script code block found in the document.")

        setup_script = setup_script_blocks[0].code

        # Extract client and tools from the setup script
        local_vars = {}
        global_vars = {"get_code_block": self.get_code_block}

        # Try to get the last expression to evaluate
        last_expr, remaining_code = self._get_last_expression(setup_script)

        exec(remaining_code, global_vars, local_vars)

        if last_expr:
            result = eval(last_expr, global_vars, local_vars)
        else:
            result = None

        client = local_vars.get("client")
        tools = local_vars.get("tools")
        self.tool_executors = local_vars.get("tool_executors", [])

        if client is None:
            raise ValueError("The 'client' variable was not defined in the setup-script.")
        if tools is None:
            raise ValueError("The 'tools' variable was not defined in the setup-script.")

        # Start tool executors
        for executor in self.tool_executors:
            if hasattr(executor, "start"):
                executor.start()

        if self.doc.conversation:
            system_message = self.doc.conversation[0].content
            if system_message is None or len(system_message) == 0:
                system_message = "You are a helpful AI assistant. "
        else:
            system_message = "You are a helpful AI assistant. "

        self.agent = ResumeFunCallAssistantAgent(
            "assistant",
            model_client=client,
            system_message=system_message,
            model_client_stream=True,
            tools=tools,
        )
        if self.doc.session_config:
            await self.agent.load_state(self.doc.session_config)

    def get_code_block(self, name: str):
        for block in self.doc.code_blocks:
            # The name of the code block is in the `attrs` field.
            if block.attrs == name:
                return block.code
        return None
