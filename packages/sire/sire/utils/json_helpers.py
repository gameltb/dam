import enum
import inspect
import json
import logging
import os
import pathlib
import sys
import typing
from collections import defaultdict
from dataclasses import asdict, fields, is_dataclass
from typing import (
    Any,
    Optional,
    Type,
    TypeVar,
    Union,
)

import torch

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _json_custom_default_encoder(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, torch.device):
        return {"type": obj.type, "index": obj.index}
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, defaultdict):
        return dict(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)  # e.g., "torch.float32"
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable: {obj}")


def _reconstruct_from_data(data: Any, expected_type: Type[T]) -> T:
    if data is None:
        return None  # type: ignore

    origin_type = typing.get_origin(expected_type)
    args_type = typing.get_args(expected_type)

    if origin_type is Union:
        if type(None) in args_type and data is None:
            return None  # type: ignore

        non_none_types = [t for t in args_type if t is not type(None)]
        if len(non_none_types) == 1:
            return _reconstruct_from_data(data, non_none_types[0])

        for member_type in non_none_types:
            try:
                reconstructed = _reconstruct_from_data(data, member_type)
                if is_dataclass(member_type) and isinstance(reconstructed, member_type):
                    return reconstructed  # type: ignore
                if not is_dataclass(member_type) and isinstance(reconstructed, member_type):
                    return reconstructed  # type: ignore
                if reconstructed is data and isinstance(data, member_type):
                    return reconstructed  # type: ignore
            except (TypeError, ValueError, AttributeError):
                continue
        logger.warning(
            f"Could not definitively reconstruct Union type {expected_type} from data: {str(data)[:100]}. Returning raw data as fallback."
        )
        return data  # Fallback

    if is_dataclass(expected_type) and isinstance(data, dict):
        kwargs = {}
        try:
            # Provide global namespace of the module where expected_type is defined for get_type_hints
            module_globals = sys.modules[expected_type.__module__].__dict__
            field_type_hints = typing.get_type_hints(expected_type, globalns=module_globals)
        except Exception as e_th:
            logger.warning(
                f"Could not get precise type hints for dataclass {expected_type.__name__}: {e_th}. Falling back to basic field types."
            )
            field_type_hints = {f.name: f.type for f in fields(expected_type)}

        for f_obj in fields(expected_type):
            field_name = f_obj.name
            actual_field_type = field_type_hints.get(field_name, f_obj.type)  # Fallback to f_obj.type if not in hints
            if field_name in data:
                reconstructed_value = _reconstruct_from_data(data[field_name], actual_field_type)
                kwargs[field_name] = reconstructed_value
        try:
            return expected_type(**kwargs)  # type: ignore[return-value]
        except Exception as e_dc_init:
            logger.error(
                f"Failed to instantiate dataclass {expected_type.__name__} with kwargs from JSON: {kwargs}. Error: {e_dc_init}",
                exc_info=True,
            )
            raise TypeError(f"Dataclass {expected_type.__name__} instantiation failed.") from e_dc_init

    elif origin_type is list and isinstance(data, list):
        if not args_type:
            return data  # type: ignore Plain list
        element_type = args_type[0]
        return [_reconstruct_from_data(item, element_type) for item in data]  # type: ignore

    elif origin_type is dict and isinstance(data, dict):
        if not args_type or len(args_type) != 2:
            return data  # type: ignore Plain dict
        _key_type, value_type = args_type
        return {key: _reconstruct_from_data(val, value_type) for key, val in data.items()}  # type: ignore

    elif expected_type is torch.device and isinstance(data, dict) and "type" in data:
        return torch.device(data["type"], data.get("index"))

    elif expected_type is pathlib.Path and isinstance(data, str):
        return pathlib.Path(data)  # type: ignore

    elif inspect.isclass(expected_type) and issubclass(expected_type, enum.Enum):
        return expected_type(data)

    elif isinstance(data, str) and expected_type is torch.dtype:
        try:
            dtype_name = data.split(".")[-1]
            if not hasattr(torch, dtype_name):  # e.g. if "float16" but torch uses "float16"
                if dtype_name == "float16":
                    dtype_name = "half"  # common alternative name
                elif dtype_name == "float32":
                    dtype_name = "float"
                elif dtype_name == "float64":
                    dtype_name = "double"
                # Add more mappings if necessary
            return getattr(torch, dtype_name)
        except AttributeError:
            logger.error(f"Could not convert string '{data}' to torch.dtype.")
            raise TypeError(f"Cannot convert '{data}' to torch.dtype")

    if expected_type is Any or isinstance(data, expected_type):  # type: ignore
        return data

    try:  # Attempt direct coercion for simple types (e.g. "1" -> 1 if int expected)
        return expected_type(data)  # type: ignore
    except (TypeError, ValueError):
        logger.debug(
            f"Data type {type(data)} for value '{str(data)[:50]}' not instance of {expected_type} and direct coercion failed. Returning as is."
        )
        return data  # type: ignore


def save_to_json_file(data_object: Any, filepath: str):
    try:
        base_dir = os.path.dirname(filepath)
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data_object, f, indent=4, default=_json_custom_default_encoder)
        logger.info(f"Data saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save data to {filepath}: {e}", exc_info=True)
        raise


def load_from_json_file(filepath: str, expected_type: Type[T]) -> Optional[T]:
    if not os.path.exists(filepath):
        logger.debug(f"File not found for loading: {filepath}")
        return None
    try:
        with open(filepath, "r") as f:
            raw_data_from_json = json.load(f)
        reconstructed_instance = _reconstruct_from_data(raw_data_from_json, expected_type)
        logger.info(
            f"Data loaded and reconstructed from {filepath} as type {expected_type.__name__ if hasattr(expected_type, '__name__') else str(expected_type)}."
        )
        return reconstructed_instance
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error loading {filepath}: {e}", exc_info=True)
    except (TypeError, ValueError) as e:  # Errors from _reconstruct_from_data
        logger.error(f"Type reconstruction error for {expected_type} from {filepath}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Failed to load/reconstruct {filepath} as {expected_type}: {e}", exc_info=True)
    return None
