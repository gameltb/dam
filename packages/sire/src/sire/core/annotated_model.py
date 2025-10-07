"""Models for annotated types."""

import logging
from typing import Any, TypeVar

from pydantic import BaseModel

_logger = logging.getLogger(__name__)


class AnnotatedBaseModel(BaseModel):
    """Base model for annotated types."""


M = TypeVar("M", bound=AnnotatedBaseModel)


def find_annotated_model[M: AnnotatedBaseModel](annotation: Any, model_type: type[M] = AnnotatedBaseModel) -> M | None:  # type: ignore
    """
    Find an annotated model in a type annotation.

    Args:
        annotation: The type annotation to search.
        model_type: The type of model to find.

    Returns:
        The annotated model, or None if not found.

    """
    if isinstance(annotation, model_type):
        return annotation
    if hasattr(annotation, "__metadata__"):
        annotated_model = None
        build_kwargs_list: list[slice] = []
        for meta in reversed(annotation.__metadata__):
            if isinstance(meta, model_type):
                annotated_model = meta
                break
            if type(meta) is slice:
                build_kwargs_list.append(meta)
        if annotated_model is not None:
            build_kwargs: dict[str, Any] = {}
            for build_kwarg in reversed(build_kwargs_list):
                if isinstance(build_kwarg.start, str):
                    build_kwargs[build_kwarg.start] = build_kwarg.stop

            build_kwargs_kset = set(build_kwargs.keys())
            model_fields_set = set(type(annotated_model).model_fields.keys())
            for k in build_kwargs_kset - model_fields_set:
                _logger.warning("Ignored annotated key %s:%s.", k, build_kwargs[k])

            annotated_model = annotated_model.model_copy(update=build_kwargs)
        return annotated_model
    return None
