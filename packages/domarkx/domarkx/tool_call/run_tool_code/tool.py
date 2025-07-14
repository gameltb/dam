import io
import logging

from rich.console import Console

from ...utils.no_border_rich_tracebacks import NoBorderTraceback

REGISTERED_TOOLS = {}


# 初始化一个Console实例，输出到StringIO以捕获内容
console_output = io.StringIO()
console = Console(file=console_output, soft_wrap=True)


def register_tool(name: str):
    """
    一个装饰器，用于注册一个函数作为可执行工具。

    参数:
        name (str): 工具的名称（对应于XML标签名）。
    """

    def decorator(func):
        REGISTERED_TOOLS[name] = func
        return func

    return decorator


def execute_tool_call(tool_call: dict, return_traceback=False, handle_exception=True):
    """
    执行一个解析出的工具调用。

    参数:
        tool_call (dict): 由 parse_tool_calls 返回的单个工具调用字典。

    返回:
        tuple: (tool_name: str, result: str) 工具执行的结果。

    抛出:
        ValueError: 如果找不到对应的工具。
        TypeError: 如果参数类型不正确。
        RuntimeError: 如果工具执行时发生其他错误。
    """
    tool_name = tool_call.get("tool_name")
    parameters = tool_call.get("parameters", {})

    logging.info(f"正在尝试执行工具 '{tool_name}'，参数: {parameters}")

    if tool_name not in REGISTERED_TOOLS:
        error_msg = f"未找到工具 '{tool_name}'。"
        logging.error(error_msg)
        raise ValueError(error_msg)

    tool_func = REGISTERED_TOOLS[tool_name]

    try:
        result = tool_func(**parameters)
        logging.info(f"工具 '{tool_name}' 执行成功。")
        return tool_name, str(result)  # 确保返回工具名和字符串结果
    except Exception as e:
        error_msg = f"执行工具 '{tool_name}' 时发生错误: {e}"
        logging.error(error_msg, exc_info=True)  # exc_info=True 会在日志中包含异常信息
        if not handle_exception:
            raise e
        console_output.truncate(0)
        console_output.seek(0)
        console.print(NoBorderTraceback(show_locals=True, extra_lines=1, max_frames=1))
        traceback_str = console_output.getvalue()
        return (
            tool_name,
            f"Error : {error_msg}\n" + (f"Traceback:\n{'\n'.join([line.rstrip() for line in traceback_str.splitlines()])}" if return_traceback else ""),
        )


def format_assistant_response(tool_name: str, tool_result: str) -> str:
    """
    将工具执行结果格式化为助手的完整回复。
    """
    return f'<tool_output tool_name="{tool_name}">\n{tool_result}\n</tool_output>'


def register_tools():
    from ...tools import (
        apply_diff,
        attempt_completion,
        execute_command,
        list_files,
        read_file,
        write_to_file,
    )

    register_tool("apply_diff")(apply_diff.apply_diff_tool)
    register_tool("attempt_completion")(attempt_completion.attempt_completion_tool)
    register_tool("execute_command")(execute_command.execute_command)
    register_tool("list_files")(list_files.list_files)
    register_tool("read_file")(read_file.read_file)
    register_tool("write_to_file")(write_to_file.write_to_file)


register_tools()
