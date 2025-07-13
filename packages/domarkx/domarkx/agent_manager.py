from domarkx.agents.resume_funcall_assistant_agent import ResumeFunCallAssistantAgent
from domarkx.tools.execute_command import execute_command
from domarkx.tools.list_files import list_files
from domarkx.tools.python_code_handler import python_code_handler
from domarkx.tools.read_file import read_file
from domarkx.tools.write_to_file import write_to_file

async def create_agent(client, system_message, chat_agent_state):
    agent = ResumeFunCallAssistantAgent(
        "assistant",
        model_client=client,
        system_message=system_message,
        model_client_stream=True,
        tools=[list_files, read_file, write_to_file, execute_command, python_code_handler],
    )
    await agent.load_state(chat_agent_state)
    return agent
