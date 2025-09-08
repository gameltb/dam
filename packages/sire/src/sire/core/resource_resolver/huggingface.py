import logging
from typing import Optional
from urllib.parse import ParseResult

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

# from ...common.path_tool import get_local_huggingface_path
from ..resource_management import (
    GLOBAL_RESOURCE_RESOLVER,
    FileSystemResource,
    Resource,
    ResourceResolver,
)

_logger = logging.getLogger(__name__)


class HuggingfaceResourceResolver(ResourceResolver):
    def _get_resource(self, url: ParseResult) -> Optional[Resource]:
        repo_id = url.path.strip("/")

        # local_path = get_local_huggingface_path(repo_id)
        # if os.path.exists(local_path):
        #     return FileSystemResource(local_path)

        # for git_dir in []:
        #     local_path = os.path.join(git_dir, repo_id)
        #     if os.path.exists(local_path):
        #         return FileSystemResource(local_path)

        try:
            local_path = snapshot_download(repo_id, local_files_only=True)
            return FileSystemResource(local_path)
        except LocalEntryNotFoundError as e:
            _logger.warning(e, exc_info=True)

        local_path = snapshot_download(repo_id, max_workers=1)
        return FileSystemResource(local_path)


GLOBAL_RESOURCE_RESOLVER.registe_scheme_resolver("huggingface", HuggingfaceResourceResolver())


def hf(repo_id: str, repo_type: Optional[str] = None) -> str:
    # repo_type is not used
    resource = GLOBAL_RESOURCE_RESOLVER.get_resource(ParseResult("huggingface", "", repo_id, "", "", ""))
    if resource is None:
        raise FileNotFoundError(f"huggingface resource not found: {repo_id}")
    path = resource.get_filesystem_path()
    if path is None:
        raise FileNotFoundError(f"huggingface resource not found: {repo_id}")
    return str(path)
