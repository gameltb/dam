from pathlib import Path

import pytest

from dam_fs.utils.url_utils import get_local_path_for_url


def test_get_local_path_for_url_file_scheme() -> None:
    # Test with a simple file URI
    url = "file:///tmp/test_file.txt"
    expected_path = Path("/tmp/test_file.txt")
    assert get_local_path_for_url(url) == expected_path

    # Test with a file URI with a hostname (should be ignored)
    url = "file://localhost/tmp/test_file.txt"
    expected_path = Path("/tmp/test_file.txt")
    assert get_local_path_for_url(url) == expected_path


def test_get_local_path_for_url_unsupported_scheme() -> None:
    # Test with an unsupported scheme
    url = "http://example.com/some_file"
    with pytest.raises(ValueError, match="Unsupported URL scheme for local access: 'http://'"):
        get_local_path_for_url(url)
