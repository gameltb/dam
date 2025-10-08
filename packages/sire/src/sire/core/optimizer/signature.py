"""Generates unique signatures for model configurations and optimization plans."""

import hashlib
import logging
import sys
from collections.abc import Callable
from enum import Enum
from typing import Any, cast

import torch
from torch import nn

logger = logging.getLogger(__name__)

MAX_STR_LEN_SIG = 30
MAX_STR_LEN_RAW = 100


class SignatureType(Enum):
    """Enum for the different levels of signature generation."""

    INPUT_ONLY = "input_only"  # Basic info and input shapes.
    WITH_WEIGHT_SHAPES = "with_weight_shapes"  # Adds model structure and weight shapes.
    WITH_WEIGHT_HASH = "with_weight_hash"  # Adds weight value hashes.


class ConfigSignatureGenerator:
    """
    Generate a unique signature for a model's configuration.

    The signature includes its architecture, weights, and input shapes.
    This signature is used to cache optimization plans.
    """

    def _serialize_value_recursive(  # noqa: PLR0912
        self,
        val: Any,
        path: list[str],
        sig_parts: list[str],
        raw: dict[str, Any],
        md: int = 3,
        cd: int = 0,
        mc: int = 5,
    ) -> None:
        if cd > md:
            sig_parts.append(f"{'_'.join(path)}_depthlimit")
            raw["status"] = "depth_limit"
            return
        ps = "_".join(path)
        if isinstance(val, torch.Tensor):
            s, dt = "_".join(map(str, val.shape)), str(val.dtype).split(".")[-1]
            sig_parts.append(f"{ps}_T_s{s}_dt{dt}")
            raw.update({"type": "Tensor", "shape": list(val.shape), "dtype": str(val.dtype)})
        elif isinstance(val, (list, tuple)):
            tc = "L" if isinstance(val, list) else "Tu"
            sig_parts.append(f"{ps}_{tc}_len{len(val)}")  # type: ignore
            raw.update({"type": val.__class__.__name__, "len": len(val), "elements": []})  # type: ignore
            elements: list[dict[str, Any]] = []
            raw["elements"] = elements
            iterable_val = cast(list[Any] | tuple[Any, ...], val)
            for i, item in enumerate(iterable_val):
                if i >= mc:
                    sig_parts.append(f"{ps}_{i}_itemlimit")
                    elements.append({"status": "item_limit"})
                    break
                inode: dict[str, Any] = {}
                elements.append(inode)
                self._serialize_value_recursive(item, [*path, str(i)], sig_parts, inode, md, cd + 1, mc)
        elif isinstance(val, dict):
            sig_parts.append(f"{ps}_D_len{len(val)}")  # type: ignore
            raw.update({"type": "Dict", "len": len(val), "items": {}})  # type: ignore
            try:
                dict_val = cast(dict[Any, Any], val)
                keys: list[Any] = sorted(list(dict_val.keys()), key=str)
            except TypeError:
                logger.debug("Dict keys %s not sortable by str.", ps)
                keys = list(val.keys())  # type: ignore
            for i, k_ in enumerate(keys):
                if i >= mc:
                    sig_parts.append(f"{ps}_{k_!s}_itemlimit")
                    raw["items"][str(k_)] = {"status": "item_limit"}
                    break
                inode = {}
                raw["items"][str(k_)] = inode
                self._serialize_value_recursive(val[k_], [*path, str(k_)], sig_parts, inode, md, cd + 1, mc)
        elif isinstance(val, (int, float, bool, str)):
            sval = str(val)
            sval_sig = (sval[:27] + "...") if isinstance(val, str) and len(sval) > MAX_STR_LEN_SIG else sval
            sig_parts.append(f"{ps}_V_{sval_sig}")
            raw.update(
                {"type": val.__class__.__name__, "value": (sval[:97] + "...") if len(sval) > MAX_STR_LEN_RAW else sval}
            )
        elif val is None:
            sig_parts.append(f"{ps}_None")
            raw["type"] = "NoneType"
        else:
            tn = val.__class__.__name__
            sig_parts.append(f"{ps}_O_{tn}")
            raw["type"] = tn
            try:
                raw["value_str"] = str(val)[:100]
            except Exception:
                raw["value_str"] = "Error_str_conversion"

    def _get_input_parts(self, args: tuple[Any, ...], kwargs: dict[str, Any], raw_in: dict[str, Any]) -> list[str]:
        s_p: list[str] = []
        args_list: list[dict[str, Any]] = []
        raw_in["args"] = args_list
        for i, v_ in enumerate(args):
            node: dict[str, Any] = {}
            args_list.append(node)
            self._serialize_value_recursive(v_, [f"arg{i}"], s_p, node)
        kwargs_dict: dict[str, Any] = {}
        raw_in["kwargs"] = kwargs_dict
        for k_, v_ in sorted(kwargs.items()):
            node: dict[str, Any] = {}
            kwargs_dict[k_] = node
            self._serialize_value_recursive(v_, [f"kw_{k_}"], s_p, node)
        return s_p

    def _get_weights_parts(self, mod: nn.Module, raw_w: dict[str, Any], include_hash: bool) -> list[str]:
        s_p: list[str] = []
        raw_w.update({"parameters": {}, "buffers": {}})
        for n, p_ in sorted(mod.named_parameters(recurse=True), key=lambda x: x[0]):
            s, dt = "_".join(map(str, p_.shape)), str(p_.dtype).split(".")[-1]
            part = f"p_{n}_s{s}_dt{dt}"
            raw_w["parameters"][n] = {"shape": list(p_.shape), "dtype": str(p_.dtype)}
            if include_hash:
                val_hash = p_.sum().item()
                part += f"_v{val_hash}"
                raw_w["parameters"][n]["val_hash"] = val_hash
            s_p.append(part)
        for n, b_ in sorted(mod.named_buffers(recurse=True), key=lambda x: x[0]):
            s, dt = "_".join(map(str, b_.shape)), str(b_.dtype).split(".")[-1]
            part = f"b_{n}_s{s}_dt{dt}"
            raw_w["buffers"][n] = {"shape": list(b_.shape), "dtype": str(b_.dtype)}
            if include_hash:
                val_hash = b_.sum().item()
                part += f"_v{val_hash}"
                raw_w["buffers"][n]["val_hash"] = val_hash
            s_p.append(part)
        return s_p

    def generate_config_signature(
        self,
        mod: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        dtype: torch.dtype,
        cb: Callable[..., Any] | None = None,
        level: SignatureType = SignatureType.WITH_WEIGHT_HASH,
    ) -> tuple[str, dict[str, Any]]:
        """Generate a signature for the given model configuration."""
        class_name = f"{mod.__class__.__module__}.{mod.__class__.__name__}"
        raw: dict[str, Any] = {
            "inputs": {},
            "module_structure": {"class_name": class_name},
            "weights": {},
            "config": {},
        }
        s_p: list[str] = [f"cls_{class_name}"]

        # Level 1: Inputs
        s_p.extend(self._get_input_parts(args, kwargs, raw["inputs"]) or ["inputs_empty"])

        # Level 2 & 3: Model structure, weights
        if level in [SignatureType.WITH_WEIGHT_SHAPES, SignatureType.WITH_WEIGHT_HASH]:
            include_weight_hash = level == SignatureType.WITH_WEIGHT_HASH
            w_parts = self._get_weights_parts(mod, raw["weights"], include_hash=include_weight_hash)
            if w_parts:
                s_p.append(f"w_{hashlib.md5('_'.join(w_parts).encode()).hexdigest()[:16]}")
                raw["weights"]["hash"] = s_p[-1]
            else:
                s_p.append("w_empty")
        else:
            raw.pop("weights")
            raw.pop("module_structure")

        if cb:
            try:
                custom = cb(mod, args, kwargs)
                raw["config"]["custom_cb_out"] = custom
                if isinstance(custom, dict):
                    s_p.extend([f"cc_{k_}{v_}" for k_, v_ in sorted(custom.items())])  # type: ignore
                else:
                    logger.warning("Custom sig cb no dict.")
            except Exception:
                logger.exception("Custom sig cb fail")
                raw["config"]["custom_cb_out"] = {"error": "exception"}

        s_p.append(f"moddt_{str(dtype).split('.')[-1]}")
        raw["config"]["mod_dtype"] = str(dtype)
        s_p.append(f"py_{sys.version_info.major}.{sys.version_info.minor}")
        raw["config"]["py_ver"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        s_p.append(f"pt_{torch.__version__}")
        raw["config"]["pt_ver"] = torch.__version__

        full_sig = "_".join(s_p)
        raw["full_sig_unhashed"] = full_sig
        hashed_sig = hashlib.md5(full_sig.encode()).hexdigest()
        raw["final_config_sig_hash"] = hashed_sig
        logger.debug("Config sig hash: %s (from: %s...)", hashed_sig, full_sig[:100])
        return hashed_sig, raw

    def generate_plan_identifier(self, mem_bytes: dict[str, int]) -> str:
        """Generate a unique identifier for an optimization plan."""
        parts = [f"{k}{v / (1024**3):.1f}G" for k, v in sorted(mem_bytes.items())]
        mem_str = "_".join(parts) or "mem_auto_empty"
        plan_id = hashlib.md5(mem_str.encode()).hexdigest()[:16]
        logger.debug("Plan ID: %s (mem: %s)", plan_id, mem_str)
        return plan_id
