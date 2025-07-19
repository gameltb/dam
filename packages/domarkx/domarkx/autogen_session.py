import ast
import json
import pathlib

from autogen_ext.models._utils.parse_r1_content import parse_r1_content

from domarkx.agents.resume_funcall_assistant_agent import ResumeFunCallAssistantAgent
from domarkx.session import Session
from domarkx.utils.chat_doc_parser import CodeBlock, Message


class AutoGenSession(Session):
    def __init__(self, doc_path: pathlib.Path):
        super().__init__(doc_path)
        self.messages = []
        self.system_message = ""

    @classmethod
    def create_message(cls, speaker: str, content: str, metadata: dict) -> "Message":
        """
        Creates a Message object with metadata as a code block.

        Args:
            speaker (str): The speaker of the message.
            content (str): The content of the message.
            metadata (dict): The metadata of the message.

        Returns:
            Message: The created Message object.
        """
        metadata_code_block = CodeBlock(
            language="json", attrs="msg-metadata", code=json.dumps(metadata, indent=2, ensure_ascii=False)
        )
        return Message(speaker=speaker, content=content, code_blocks=[metadata_code_block])

    def _get_last_expression(self, code):
        try:
            tree = ast.parse(code)
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = tree.body.pop()
                return ast.unparse(last_expr), ast.unparse(tree)
            return None, code
        except SyntaxError:
            return None, code

    def _process_initial_messages(self):
        if self.doc.conversation:
            self.system_message = self.doc.conversation[0].content
            if self.system_message is None or len(self.system_message) == 0:
                self.system_message = "You are a helpful AI assistant. "
        else:
            self.system_message = "You are a helpful AI assistant. "

        messages = []
        for md_message in self.doc.conversation[1:]:
            message_dict = md_message.metadata
            thought, content = parse_r1_content(md_message.content)
            if "content" not in message_dict:
                message_dict["content"] = content
            elif isinstance(message_dict["content"], list) and len(message_dict["content"]) == 1:
                if "content" not in message_dict["content"][0] and "arguments" not in message_dict["content"][0]:
                    message_dict["content"][0]["content"] = content
            if thought:
                message_dict["thought"] = "\n".join(line.removeprefix("> ") for line in thought.splitlines())
            messages.append(message_dict)
        self.messages = messages

    def append_new_messages(self, new_state: dict):
        from domarkx.utils.chat_doc_parser import append_message

        for message in new_state["llm_context"]["messages"][len(self.messages) :]:
            content = ""
            if "content" in message:
                if isinstance(message["content"], str):
                    content = message.pop("content")
                elif isinstance(message["content"], list) and len(message["content"]) == 1:
                    content = message["content"][0].pop("content", "")
            thought = message.pop("thought", "")
            if thought:
                thought = "\n".join("> " + line for line in f"""<think>{thought}</think>""".splitlines())
                content = f"""
{thought}

{content}"""
            with self.doc_path.open("a") as f:
                append_message(f, self.create_message("assistant", content, message))

    async def setup(self, **kwargs):
        self._process_initial_messages()
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

        self.agent = ResumeFunCallAssistantAgent(
            "assistant",
            model_client=client,
            system_message=self.system_message,
            model_client_stream=True,
            tools=tools,
        )
        if self.doc.session_config:
            agent_state = self.doc.session_config
            agent_state["llm_context"]["messages"] = self.messages
            await self.agent.load_state(agent_state)
