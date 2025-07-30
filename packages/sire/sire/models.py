from typing import Any, Optional

from pydantic import BaseModel


class ModelConfig(BaseModel):
    name: str
    model_path: str
    runtime: str
    model_class: Optional[Any] = None
    # To store the loaded model instance
    model: Optional[Any] = None
    # To store the device it's loaded on
    device: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
