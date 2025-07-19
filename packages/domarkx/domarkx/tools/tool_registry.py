import importlib
import inspect
import os
from typing import Any, Callable, Dict

from domarkx.tools.tool_decorator import tool_handler

_tool_registry: Dict[str, Callable[..., Any]] = {}
_tool_registry_unwrapped: Dict[str, Callable[..., Any]] = {}


def _discover_and_register_tools():
    """
    Dynamically discovers and registers tools from the 'portable_tools' directory.
    """
    global _tool_registry, _tool_registry_unwrapped
    if _tool_registry:
        return

    portable_tools_path = os.path.join(os.path.dirname(__file__), "portable_tools")
    for root, _, files in os.walk(portable_tools_path):
        for file in files:
            if file.endswith(".py") and not file.startswith("__init__"):
                module_path = os.path.join(root, file)
                module_name = os.path.splitext(os.path.relpath(module_path, portable_tools_path))[0].replace(
                    os.sep, "."
                )
                module = importlib.import_module(f"domarkx.tools.portable_tools.{module_name}")

                with open(module_path, "r") as f:
                    module_source = f.read()

                for attribute_name, attribute in inspect.getmembers(module, inspect.isfunction):
                    if attribute_name.startswith("tool_"):
                        # Store the original, unwrapped function
                        _tool_registry_unwrapped[attribute_name] = attribute

                        # Apply the decorator and store the wrapped function
                        wrapped_tool = tool_handler()(attribute)
                        wrapped_tool.__source_code__ = module_source
                        _tool_registry[attribute_name] = wrapped_tool


def get_tool(name: str) -> Callable[..., Any]:
    """
    Retrieves a tool from the registry by its name.

    Args:
        name (str): The name of the tool to retrieve.

    Returns:
        Callable[..., Any]: The wrapped tool function.

    Raises:
        KeyError: If the tool is not found.
    """
    return _tool_registry[name]


def get_unwrapped_tool(name: str) -> Callable[..., Any]:
    """
    Retrieves an unwrapped tool from the registry by its name.

    Args:
        name (str): The name of the tool to retrieve.

    Returns:
        Callable[..., Any]: The unwrapped tool function.

    Raises:
        KeyError: If the tool is not found.
    """
    return _tool_registry_unwrapped[name]


def list_available_tools():
    """
    Lists all available tool functions registered in the tool registry.

    Returns:
        List[dict]: Each dict contains 'name', 'doc', and 'signature' for a tool.
    """
    tool_infos = []
    for name, func in _tool_registry.items():
        tool_infos.append(
            {
                "name": name,
                "doc": inspect.getdoc(func),
                "signature": str(inspect.signature(func)),
            }
        )
    return tool_infos


# Discover and register tools on module import
_discover_and_register_tools()

# To maintain backwards compatibility with the old list_available_tools,
# we'll register it as a tool.
_tool_registry["list_available_tools"] = tool_handler()(list_available_tools)
_tool_registry_unwrapped["list_available_tools"] = list_available_tools
