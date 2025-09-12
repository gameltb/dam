# type: ignore
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Optional, Union
from urllib.parse import ParseResult, urlparse

_logger = logging.getLogger(__name__)


class Resource:
    def get_bytes_io(self) -> BinaryIO:
        raise NotImplementedError()

    def get_filesystem_path(self) -> Optional[Path]:
        raise NotImplementedError()


class FileSystemResource(Resource):
    def __init__(self, file_path: Union[str, Path]) -> None:
        self.file_path = file_path

    def get_bytes_io(self) -> BinaryIO:
        return open(self.file_path, "rb")

    def get_filesystem_path(self) -> Optional[Path]:
        return Path(self.file_path)


class BytesBuffResource(Resource):
    def __init__(self, bytes_buff: bytes) -> None:
        self.bytes_buff = bytes_buff

    def get_bytes_io(self) -> BytesIO:
        return BytesIO(self.bytes_buff)


class ResourceResolver:
    def get_resource(
        self, url: Union[str, ParseResult], cls: Optional[type["ResourceResolver"]] = None
    ) -> Optional[Resource]:
        _logger.info(f"try get resource {url}")
        if isinstance(url, str):
            url = urlparse(url)

        if cls is None:
            cls = self.__class__

        if (resource := cls._get_resource(self, url)) is not None:
            return resource
        else:
            raise FileNotFoundError()

    def _get_resource(self, url: ParseResult) -> Optional[Resource]:
        raise NotImplementedError()


class MutiSchemeResourceResolver(ResourceResolver):
    def __init__(self) -> None:
        self.scheme_resolver_map: dict[str, ResourceResolver] = {}

    def registe_scheme_resolver(self, scheme: str, scheme_resolver: "ResourceResolver"):
        self.scheme_resolver_map[scheme] = scheme_resolver

    def _get_resource(self, url: ParseResult) -> Optional[Resource]:
        scheme_resolver = self.scheme_resolver_map[url.scheme]
        return scheme_resolver._get_resource(url)


class ProxyMutiSchemeResourceResolver(MutiSchemeResourceResolver):
    def __init__(self) -> None:
        super().__init__()
        self.identical_resource_url_map: dict[str, set[str]] = {}

    def registe_identical_resource_urls(self, identical_resource_urls: set[str]):
        new_identical_resource_urls = set()
        for url in identical_resource_urls:
            new_identical_resource_urls.add(url)
            if old_identical_resource_urls := self.identical_resource_url_map.get(url):
                new_identical_resource_urls.update(old_identical_resource_urls)
        for url in new_identical_resource_urls:
            self.identical_resource_url_map[url] = new_identical_resource_urls

    def find_identical_resource_url(self, url: str, scheme: Optional[str] = None) -> Optional[set[str]]:
        if identical_resource_urls := self.identical_resource_url_map.get(url):
            if scheme is None:
                return identical_resource_urls
            new_identical_resource_urls = set()
            for id_url in identical_resource_urls:
                id_url_parse = urlparse(id_url)
                if id_url_parse.scheme == scheme:
                    new_identical_resource_urls.add(id_url)
            return new_identical_resource_urls
        return None

    def _get_resource(self, url: ParseResult) -> Optional[Resource]:
        if url.geturl() not in self.identical_resource_url_map:
            return super()._get_resource(url)

        identical_resource_urls = self.identical_resource_url_map[url.geturl()]

        for identical_url in identical_resource_urls:
            try:
                return self.get_resource(identical_url, cls=MutiSchemeResourceResolver)
            except FileNotFoundError:
                pass


GLOBAL_RESOURCE_RESOLVER = ProxyMutiSchemeResourceResolver()


class FileSchemeResourceResolver(ResourceResolver):
    def _get_resource(self, url: ParseResult) -> Optional[Resource]:
        file_path = url.path
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileSystemResource(file_path)


GLOBAL_RESOURCE_RESOLVER.registe_scheme_resolver("file", FileSchemeResourceResolver())


if __name__ == "__main__":
    GLOBAL_RESOURCE_RESOLVER.registe_identical_resource_urls(["sha256://jdie", "http://1"])
    GLOBAL_RESOURCE_RESOLVER.registe_identical_resource_urls(["http://2", "http://1"])
    print(GLOBAL_RESOURCE_RESOLVER.identical_resource_url_map)
    with GLOBAL_RESOURCE_RESOLVER.get_resource("file:////module/core/precision.py").get_bytes_io() as f:
        print(f.read(15))
