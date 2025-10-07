"""Resource management for Sire."""

# type: ignore
import logging
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from urllib.parse import ParseResult, urlparse

_logger = logging.getLogger(__name__)


class Resource:
    """An abstract base class for a resource."""

    def get_bytes_io(self) -> BinaryIO:
        """Get a binary IO stream for the resource."""
        raise NotImplementedError()

    def get_filesystem_path(self) -> Path | None:
        """Get the filesystem path for the resource, if available."""
        raise NotImplementedError()


class FileSystemResource(Resource):
    """A resource that exists on the local filesystem."""

    def __init__(self, file_path: str | Path) -> None:
        """
        Initialize the resource.

        Args:
            file_path: The path to the file.

        """
        self.file_path = Path(file_path)

    def get_bytes_io(self) -> BinaryIO:
        """Get a binary IO stream for the file."""
        return self.file_path.open("rb")

    def get_filesystem_path(self) -> Path | None:
        """Get the filesystem path for the file."""
        return self.file_path


class BytesBuffResource(Resource):
    """A resource that exists in a byte buffer."""

    def __init__(self, bytes_buff: bytes) -> None:
        """
        Initialize the resource.

        Args:
            bytes_buff: The byte buffer.

        """
        self.bytes_buff = bytes_buff

    def get_bytes_io(self) -> BytesIO:
        """Get a binary IO stream for the buffer."""
        return BytesIO(self.bytes_buff)


class ResourceResolver:
    """An abstract base class for a resource resolver."""

    def get_resource(self, url: str | ParseResult, cls: type["ResourceResolver"] | None = None) -> Resource | None:
        """
        Get a resource from a URL.

        Args:
            url: The URL of the resource.
            cls: The class to use for resolution.

        Returns:
            The resource, or None if not found.

        Raises:
            FileNotFoundError: If the resource cannot be found.

        """
        _logger.info("try get resource %s", url)
        if isinstance(url, str):
            url = urlparse(url)

        if cls is None:
            cls = self.__class__

        if (resource := cls._get_resource(self, url)) is not None:
            return resource
        raise FileNotFoundError()

    def _get_resource(self, url: ParseResult) -> Resource | None:
        raise NotImplementedError()


class MutiSchemeResourceResolver(ResourceResolver):
    """A resource resolver that can handle multiple URL schemes."""

    def __init__(self) -> None:
        """Initialize the resolver."""
        self.scheme_resolver_map: dict[str, ResourceResolver] = {}

    def registe_scheme_resolver(self, scheme: str, scheme_resolver: "ResourceResolver") -> None:
        """Register a resolver for a given scheme."""
        self.scheme_resolver_map[scheme] = scheme_resolver

    def _get_resource(self, url: ParseResult) -> Resource | None:
        scheme_resolver = self.scheme_resolver_map[url.scheme]
        return scheme_resolver._get_resource(url)


class ProxyMutiSchemeResourceResolver(MutiSchemeResourceResolver):
    """A multi-scheme resolver that can handle identical resources with different URLs."""

    def __init__(self) -> None:
        """Initialize the resolver."""
        super().__init__()
        self.identical_resource_url_map: dict[str, set[str]] = {}

    def registe_identical_resource_urls(self, identical_resource_urls: set[str]) -> None:
        """Register a set of URLs as being identical resources."""
        new_identical_resource_urls: set[str] = set()
        for url in identical_resource_urls:
            new_identical_resource_urls.add(url)
            if old_identical_resource_urls := self.identical_resource_url_map.get(url):
                new_identical_resource_urls.update(old_identical_resource_urls)
        for url in new_identical_resource_urls:
            self.identical_resource_url_map[url] = new_identical_resource_urls

    def find_identical_resource_url(self, url: str, scheme: str | None = None) -> set[str] | None:
        """
        Find identical resource URLs for a given URL.

        Args:
            url: The URL to find identical resources for.
            scheme: An optional scheme to filter the results by.

        Returns:
            A set of identical resource URLs, or None if not found.

        """
        if identical_resource_urls := self.identical_resource_url_map.get(url):
            if scheme is None:
                return identical_resource_urls
            new_identical_resource_urls: set[str] = set()
            for id_url in identical_resource_urls:
                id_url_parse = urlparse(id_url)
                if id_url_parse.scheme == scheme:
                    new_identical_resource_urls.add(id_url)
            return new_identical_resource_urls
        return None

    def _get_resource(self, url: ParseResult) -> Resource | None:
        if url.geturl() not in self.identical_resource_url_map:
            return super()._get_resource(url)

        identical_resource_urls = self.identical_resource_url_map[url.geturl()]

        for identical_url in identical_resource_urls:
            try:
                return self.get_resource(identical_url, cls=MutiSchemeResourceResolver)
            except FileNotFoundError:
                pass
        return None


GLOBAL_RESOURCE_RESOLVER = ProxyMutiSchemeResourceResolver()


class FileSchemeResourceResolver(ResourceResolver):
    """A resource resolver for the file:// scheme."""

    def _get_resource(self, url: ParseResult) -> Resource | None:
        """Get a resource from a file URL."""
        file_path = Path(url.path)
        if file_path.exists() and file_path.is_file():
            return FileSystemResource(file_path)
        return None


GLOBAL_RESOURCE_RESOLVER.registe_scheme_resolver("file", FileSchemeResourceResolver())


if __name__ == "__main__":
    GLOBAL_RESOURCE_RESOLVER.registe_identical_resource_urls({"sha256://jdie", "http://1"})
    GLOBAL_RESOURCE_RESOLVER.registe_identical_resource_urls({"http://2", "http://1"})
    _logger.info(GLOBAL_RESOURCE_RESOLVER.identical_resource_url_map)
    resource = GLOBAL_RESOURCE_RESOLVER.get_resource("file:////module/core/precision.py")
    if resource:
        with resource.get_bytes_io() as f:
            _logger.info(f.read(15))
