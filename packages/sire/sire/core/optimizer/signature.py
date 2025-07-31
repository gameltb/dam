import hashlib
import logging
import sys
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ConfigSignatureGenerator:
    """
    Generates a unique signature for a model's configuration, including its
    architecture, weights, and input shapes. This signature is used to cache
    optimization plans.
    """

    def _serialize_value_recursive(self, val: Any, path: List[str], sig_parts: List[str], raw: Dict, md=3, cd=0, mc=5):
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
            sig_parts.append(f"{ps}_{tc}_len{len(val)}")
            raw.update({"type": val.__class__.__name__, "len": len(val), "elements": []})
            for i, item in enumerate(val):
                if i >= mc:
                    sig_parts.append(f"{ps}_{i}_itemlimit")
                    raw["elements"].append({"status": "item_limit"})
                    break
                inode = {}
                raw["elements"].append(inode)
                self._serialize_value_recursive(item, path + [str(i)], sig_parts, inode, md, cd + 1, mc)
        elif isinstance(val, dict):
            sig_parts.append(f"{ps}_D_len{len(val)}")
            raw.update({"type": "Dict", "len": len(val), "items": {}})
            try:
                keys = sorted(list(val.keys()), key=str)
            except TypeError:
                logger.debug(f"Dict keys {ps} not sortable by str.")
                keys = list(val.keys())
            for i, k_ in enumerate(keys):
                if i >= mc:
                    sig_parts.append(f"{ps}_{str(k_)}_itemlimit")
                    raw["items"][str(k_)] = {"status": "item_limit"}
                    break
                inode = {}
                raw["items"][str(k_)] = inode
                self._serialize_value_recursive(val[k_], path + [str(k_)], sig_parts, inode, md, cd + 1, mc)
        elif isinstance(val, (int, float, bool, str)):
            sval = str(val)
            sval_sig = (sval[:27] + "...") if isinstance(val, str) and len(sval) > 30 else sval
            sig_parts.append(f"{ps}_V_{sval_sig}")
            raw.update({"type": val.__class__.__name__, "value": (sval[:97] + "...") if len(sval) > 100 else sval})
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

    def _get_input_parts(self, mod: nn.Module, args: Tuple, kwargs: Dict, raw_in: Dict) -> List[str]:
        s_p = []
        raw_in["args"] = []
        for i, v_ in enumerate(args):
            node: Dict = {}
            raw_in["args"].append(node)
            self._serialize_value_recursive(v_, [f"arg{i}"], s_p, node)
        raw_in["kwargs"] = {}
        for k_, v_ in sorted(kwargs.items()):
            node: Dict = {}
            raw_in["kwargs"][k_] = node
            self._serialize_value_recursive(v_, [f"kw_{k_}"], s_p, node)
        return s_p

    def _get_weights_parts(self, mod: nn.Module, raw_w: Dict) -> List[str]:
        s_p = []
        raw_w.update({"parameters": {}, "buffers": {}})
        for n, p_ in sorted(mod.named_parameters(recurse=True), key=lambda x: x[0]):
            s, dt = "_".join(map(str, p_.shape)), str(p_.dtype).split(".")[-1]
            # Add a hash of the tensor's values to be sensitive to weight changes.
            # Using sum is fast and effective for change detection.
            val_hash = p_.sum().item()
            s_p.append(f"p_{n}_s{s}_dt{dt}_v{val_hash}")
            raw_w["parameters"][n] = {"shape": list(p_.shape), "dtype": str(p_.dtype), "val_hash": val_hash}
        for n, b_ in sorted(mod.named_buffers(recurse=True), key=lambda x: x[0]):
            s, dt = "_".join(map(str, b_.shape)), str(b_.dtype).split(".")[-1]
            val_hash = b_.sum().item()
            s_p.append(f"b_{n}_s{s}_dt{dt}_v{val_hash}")
            raw_w["buffers"][n] = {"shape": list(b_.shape), "dtype": str(b_.dtype), "val_hash": val_hash}
        return s_p

    def generate_config_signature(
        self, mod: nn.Module, args: Tuple, kwargs: Dict, dtype: torch.dtype, cb=None
    ) -> Tuple[str, Dict[str, Any]]:
        raw: Dict[str, Any] = {
            "inputs": {},
            "module_structure": {"class_name": mod.__class__.__name__},
            "weights": {},
            "config": {},
        }
        s_p = [f"cls_{mod.__class__.__name__}"]
        s_p.extend(self._get_input_parts(mod, args, kwargs, raw["inputs"]) or ["inputs_empty"])
        w_parts = self._get_weights_parts(mod, raw["weights"])
        if w_parts:
            s_p.append(f"w_{hashlib.md5('_'.join(w_parts).encode()).hexdigest()[:16]}")
            raw["weights"]["hash"] = s_p[-1]
        else:
            s_p.append("w_empty")
        if cb:
            try:
                custom = cb(mod, args, kwargs)
                raw["config"]["custom_cb_out"] = custom
                if isinstance(custom, dict):
                    s_p.extend([f"cc_{k_}{v_}" for k_, v_ in sorted(custom.items())])
                else:
                    logger.warning("Custom sig cb no dict.")
            except Exception as e:
                logger.error(f"Custom sig cb fail: {e}", exc_info=True)
                raw["config"]["custom_cb_out"] = {"error": str(e)}
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
        logger.debug(f"Config sig hash: {hashed_sig} (from: {full_sig[:100]}...)")
        return hashed_sig, raw

    def generate_plan_identifier(self, mem_bytes: Dict[str, int]) -> str:
        parts = [f"{k}{v / (1024**3):.1f}G" for k, v in sorted(mem_bytes.items())]
        mem_str = "_".join(parts) or "mem_auto_empty"
        plan_id = hashlib.md5(mem_str.encode()).hexdigest()[:16]
        logger.debug(f"Plan ID: {plan_id} (mem: {mem_str})")
        return plan_id
