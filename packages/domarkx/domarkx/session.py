import ast
import pathlib

from domarkx.agents.resume_funcall_assistant_agent import ResumeFunCallAssistantAgent
from domarkx.utils.chat_doc_parser import MarkdownLLMParser


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
        try:
            return parser.parse(md_content)
        except ValueError as e:
            raise ValueError(f"Error parsing document at {self.doc_path.absolute()}: {e}") from e

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
