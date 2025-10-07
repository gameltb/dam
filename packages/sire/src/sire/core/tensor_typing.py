import logging
from typing import Annotated

import torch
from torch.export import Dim

from .annotated_model import AnnotatedBaseModel, find_annotated_model

_logger = logging.getLogger(__name__)


class TensorModel(AnnotatedBaseModel):
    dims: list[type] | None = None
    dtype: str | None = None


Tensor = Annotated[torch.Tensor, TensorModel()]

DIMS = "dims"
DTYPE = "dtype"

BATCH_DIM = Dim("batch")

if __name__ in ("__main__", "<run_path>"):
    _logger.info(find_annotated_model(Annotated[Tensor, DIMS : [Dim("C", min=1)]], model_type=TensorModel))
