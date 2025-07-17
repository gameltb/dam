import inspect
from typing import Any, Callable, Dict, Optional

from autogen_core.tools import BaseTool
from pydantic import BaseModel, Field

from domarkx.tool_executors.jupyter import JupyterToolExecutor


class ToolWrapper(BaseTool):
    tool_func: Callable
    executor: JupyterToolExecutor

    class ToolSchema(BaseModel):
        pass  # Will be populated dynamically

    def __init__(self, tool_func: Callable, executor: JupyterToolExecutor):
        super().__init__()
        self.tool_func = tool_func
        self.executor = executor
        self.name = tool_func.__name__
        self.description = inspect.getdoc(tool_func) or ""

        # Dynamically create the Pydantic model for the tool's arguments
        self.ToolSchema = self._create_schema_from_func(tool_func)

    def _create_schema_from_func(self, func: Callable) -> BaseModel:
        fields = {}
        sig = inspect.signature(func)
        for param in sig.parameters.values():
            if param.name == "self":
                continue
            # Get type annotation and default value
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
            default_value = param.default if param.default != inspect.Parameter.empty else ...
            fields[param.name] = (param_type, Field(default=default_value))

        return type(f"{func.__name__}Schema", (BaseModel,), {"__annotations__": fields})

    @property
    def schema(self) -> Dict[str, Any]:
        return self.ToolSchema.model_json_schema()

    def call(self, **kwargs: Any) -> Any:
        # Validate arguments using the dynamically created Pydantic model
        validated_args = self.ToolSchema(**kwargs)
        return self.executor.execute(self.tool_func, **validated_args.dict())
