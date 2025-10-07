"""A resource resolver for playground UUID resources."""

import threading
import uuid
from urllib.parse import ParseResult, urlparse

from ..resource_management import (
    GLOBAL_RESOURCE_RESOLVER,
    BytesBuffResource,
    Resource,
    ResourceResolver,
)


class UUIDResourceResolver(ResourceResolver):
    """A resource resolver for the playground UUID scheme."""

    def _get_resource(self, url: ParseResult) -> Resource | None:
        """Get a resource from the global UUID resource pool."""
        if url.hostname is None:
            return None
        return GLOBAL_UUID_RESOURCE_POOL.get(url.hostname)


GLOBAL_RESOURCE_RESOLVER.registe_scheme_resolver("playgrounduuidres", UUIDResourceResolver())

GLOBAL_UUID_RESOURCE_POOL: dict[str, Resource] = {}
GLOBAL_UUID_RESOURCE_POOL_LOCK = threading.Lock()


def _gen_res_uuid() -> str:
    while True:
        res_uuid = uuid.uuid4().hex
        if res_uuid not in GLOBAL_UUID_RESOURCE_POOL:
            return res_uuid


def uuid_res_pool_put(resource: Resource) -> str:
    """
    Put a resource into the global UUID resource pool.

    Args:
        resource: The resource to put into the pool.

    Returns:
        The UUID of the resource.

    """
    with GLOBAL_UUID_RESOURCE_POOL_LOCK:
        res_uuid = _gen_res_uuid()
        GLOBAL_UUID_RESOURCE_POOL[res_uuid] = resource
    return res_uuid


def uuid_res_pool_registe_url(url: str) -> str:
    """
    Register a URL with the global UUID resource pool.

    If the URL is already registered, return its existing UUID. Otherwise,
    create a new UUID for the URL and return it.

    Args:
        url: The URL to register.

    Returns:
        The UUID of the resource.

    """
    res_uuid: str | None = None
    if identical_resource_urls := GLOBAL_RESOURCE_RESOLVER.find_identical_resource_url(url, scheme="playgrounduuidres"):
        res_uuid = urlparse(next(iter(identical_resource_urls))).hostname

    if res_uuid is None:
        res_uuid = uuid_res_pool_put(BytesBuffResource(b""))
        GLOBAL_RESOURCE_RESOLVER.registe_identical_resource_urls({f"playgrounduuidres://{res_uuid}", url})
    return res_uuid
