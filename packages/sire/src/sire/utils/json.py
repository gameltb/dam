import functools
import json
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, o: Any) -> Any:
        if isinstance(
            o,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(o)

        elif isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
            return float(o)

        elif isinstance(o, (np.complex_, np.complex64, np.complex128)):
            return {"real": o.real, "imag": o.imag}

        elif isinstance(o, (np.ndarray,)):
            return o.tolist()

        elif isinstance(o, (np.bool_)):
            return bool(o)

        elif isinstance(o, (np.void)):
            return None

        return super().default(o)


np_dumps = functools.partial(json.dumps, cls=NumpyEncoder)
np_dump = functools.partial(json.dump, cls=NumpyEncoder)
