"""Resource resolver for Hugging Face Hub models and datasets."""

import logging
from urllib.parse import ParseResult

from huggingface_hub import snapshot_download  # type: ignore
from huggingface_hub.errors import LocalEntryNotFoundError

from ..resource_management import (
    GLOBAL_RESOURCE_RESOLVER,
    FileSystemResource,
    Resource,
    ResourceResolver,
)

_logger = logging.getLogger(__name__)


class HuggingfaceResourceResolver(ResourceResolver):
    """A resource resolver for the huggingface:// scheme."""

    def _get_resource(self, url: ParseResult) -> Resource | None:
        """
        Get a resource from the Hugging Face Hub.

        This first tries to load the resource from the local cache, and if it's
        not found, it downloads it.

        Args:
            url: The URL of the resource.

        Returns:
            The resource, or None if it cannot be found.

        """
        repo_id = url.path.strip("/")

        try:
            local_path = snapshot_download(repo_id, local_files_only=True)
            return FileSystemResource(local_path)
        except LocalEntryNotFoundError:
            _logger.warning("Local entry not found for %s, attempting download.", repo_id)

        local_path = snapshot_download(repo_id, max_workers=1)
        return FileSystemResource(local_path)


GLOBAL_RESOURCE_RESOLVER.registe_scheme_resolver("huggingface", HuggingfaceResourceResolver())


def hf(repo_id: str) -> str:
    """
    Get the local filesystem path for a Hugging Face Hub resource.

    Args:
        repo_id: The ID of the repository on the Hugging Face Hub.

    Returns:
        The local filesystem path to the resource.

    Raises:
        FileNotFoundError: If the resource cannot be found.

    """
    resource = GLOBAL_RESOURCE_RESOLVER.get_resource(ParseResult("huggingface", "", repo_id, "", "", ""))
    if resource is None:
        raise FileNotFoundError(f"huggingface resource not found: {repo_id}")
    path = resource.get_filesystem_path()
    if path is None:
        raise FileNotFoundError(f"huggingface resource not found: {repo_id}")
    return str(path)
