"""A marker component indicating that a CSO file has been successfully ingested."""

from dam.models.core.base_component import UniqueComponent


class IngestedCsoComponent(UniqueComponent):
    """
    A marker component indicating that a CSO file has been successfully ingested.

    This also implies that a corresponding virtual ISO entity has been created.
    """

    __tablename__ = "component_ingested_cso"
