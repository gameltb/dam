import functools
import json
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, o: Any) -> Any:
        if hasattr(o, "dtype"):
            if o.dtype.kind in "iu":
                return int(o)
            elif o.dtype.kind == "f":
                return float(o)
            elif o.dtype.kind == "c":
                return {"real": o.real, "imag": o.imag}
            elif o.dtype.kind == "b":
                return bool(o)
            elif o.dtype.kind == "V":
                return None

        if isinstance(o, (np.ndarray,)):
            return o.tolist()

        return super().default(o)


np_dumps = functools.partial(json.dumps, cls=NumpyEncoder)
np_dump = functools.partial(json.dump, cls=NumpyEncoder)
