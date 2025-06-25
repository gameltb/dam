import subprocess  # Moved to top
from pathlib import Path

import pytest

from dam.services.file_operations import (
    calculate_sha256,
    generate_perceptual_hashes,
    get_file_properties,
)


# Fixture to provide paths to sample images
@pytest.fixture(scope="module")
def test_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Creates a temporary directory and populates it with test image files.
    This is better than relying on files in the repo for more hermetic tests.
    """
    data_dir = tmp_path_factory.mktemp("file_ops_test_data")

    def _create_dummy_image_for_file_ops(filepath: Path, color_name: str, size=(10, 10)):
        colors = {
            "red": (255, 0, 0),
            "blue": (0, 0, 255),
        }
        try:
            from PIL import Image

            img = Image.new("RGB", size, color=colors[color_name])
            if color_name == "blue":
                for i in range(size[0]):
                    img.putpixel((i, i), (255, 255, 255))
            filepath.parent.mkdir(parents=True, exist_ok=True)
            img.save(filepath, "PNG")
        except ImportError:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(f"dummy_image_{color_name}".encode())

    _create_dummy_image_for_file_ops(data_dir / "img_A.png", "red")
    _create_dummy_image_for_file_ops(data_dir / "img_B.png", "blue")
    return data_dir


@pytest.fixture
def sample_image_path_a(test_data_dir: Path) -> Path:
    return test_data_dir / "img_A.png"


@pytest.fixture
def sample_image_path_b(test_data_dir: Path) -> Path:
    return test_data_dir / "img_B.png"


@pytest.fixture
def non_image_file_path(tmp_path: Path) -> Path:
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("This is not an image.")
    return txt_file


def test_generate_perceptual_hashes_valid_image(sample_image_path_a: Path):
    try:
        import imagehash  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        pytest.skip("ImageHash or Pillow not installed, skipping perceptual hash generation test.")
    hashes = generate_perceptual_hashes(sample_image_path_a)
    assert isinstance(hashes, dict)
    possible_keys = {"phash", "ahash", "dhash"}
    assert any(key in hashes for key in possible_keys), "No perceptual hashes were generated."
    for key in possible_keys:
        if key in hashes:
            assert isinstance(hashes[key], str), f"{key} value is not a string."
            assert len(hashes[key]) > 0, f"{key} value is an empty string."


def test_generate_perceptual_hashes_different_images_produce_different_hashes(
    sample_image_path_a: Path, sample_image_path_b: Path
):
    try:
        import imagehash  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        pytest.skip("ImageHash or Pillow not installed, skipping perceptual hash comparison test.")
    hashes_a = generate_perceptual_hashes(sample_image_path_a)
    hashes_b = generate_perceptual_hashes(sample_image_path_b)
    if not all(k in hashes_a for k in ["phash", "ahash", "dhash"]) or not all(
        k in hashes_b for k in ["phash", "ahash", "dhash"]
    ):
        pytest.skip("Not all hash types generated for one or both sample images, cannot compare.")
    assert hashes_a["phash"] != hashes_b["phash"]
    assert hashes_a["ahash"] != hashes_b["ahash"]
    assert hashes_a["dhash"] != hashes_b["dhash"]


def test_generate_perceptual_hashes_non_image_file(non_image_file_path: Path, caplog):
    caplog.clear()
    hashes = generate_perceptual_hashes(non_image_file_path)
    assert hashes == {}
    assert any(
        record.levelname == "WARNING" and "Could not open or process image" in record.message
        for record in caplog.records
    )


def test_generate_perceptual_hashes_missing_file(tmp_path: Path, caplog):
    non_existent_file = tmp_path / "does_not_exist.png"
    caplog.clear()
    hashes = generate_perceptual_hashes(non_existent_file)
    assert hashes == {}
    assert any(
        record.levelname == "WARNING" and f"Image file not found at {non_existent_file}" in record.message
        for record in caplog.records
    )


def test_calculate_sha256_simple_file(tmp_path: Path):
    file_content = b"hello world"
    test_file = tmp_path / "sha_test.txt"
    test_file.write_bytes(file_content)
    expected_hash = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert calculate_sha256(test_file) == expected_hash


def test_get_file_properties_simple_file(tmp_path: Path):
    file_content = b"some data"
    test_file = tmp_path / "props_test.dat"
    test_file.write_bytes(file_content)
    name, size, mime = get_file_properties(test_file)
    assert name == "props_test.dat"
    assert size == len(file_content)
    assert mime is not None


@pytest.mark.parametrize(
    "mock_file_output, mock_mimetypes_guess, expected_mime, file_ext",
    [
        ("image/png", None, "image/png", ".png"),
        ("text/plain", None, "text/plain", ".txt"),
        (FileNotFoundError(), "image/jpeg", "image/jpeg", ".jpg"),
        (FileNotFoundError(), None, "application/octet-stream", ".unknown"),
        (subprocess.CalledProcessError(1, "file cmd"), "application/pdf", "application/pdf", ".pdf"),
        ("", "text/xml", "text/xml", ".xml"),
        (FileNotFoundError(), None, "application/octet-stream", ".dat"),
        ("", None, "application/octet-stream", ".foo"),
    ],
)
def test_get_file_properties_mime_detection_logic(
    tmp_path: Path, monkeypatch, mock_file_output, mock_mimetypes_guess, expected_mime, file_ext
):
    import mimetypes

    test_file = tmp_path / f"testfile{file_ext}"
    test_file.write_text("dummy content")
    mock_subprocess_run = None
    if isinstance(mock_file_output, Exception):

        def _mock_subprocess_run_exception(*args, **kwargs):
            raise mock_file_output

        mock_subprocess_run = _mock_subprocess_run_exception
    else:

        def _mock_subprocess_run_success(*args, **kwargs):
            return subprocess.CompletedProcess(args[0], 0, stdout=mock_file_output, stderr="")

        mock_subprocess_run = _mock_subprocess_run_success
    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)
    monkeypatch.setattr(mimetypes, "guess_type", lambda path: (mock_mimetypes_guess, None))
    _, _, mime_type = get_file_properties(test_file)
    assert mime_type == expected_mime


def test_get_file_properties_file_command_not_present_logs_warning(tmp_path: Path, monkeypatch, caplog):
    import mimetypes

    test_file = tmp_path / "test.txt"
    test_file.write_text("content")

    def mock_run_file_not_found(*args, **kwargs):
        raise FileNotFoundError("'file' command not found")

    monkeypatch.setattr(subprocess, "run", mock_run_file_not_found)
    monkeypatch.setattr(mimetypes, "guess_type", lambda path: ("text/plain_fallback", None))
    caplog.clear()
    _, _, mime = get_file_properties(test_file)
    assert mime == "text/plain_fallback"
    assert any(
        record.levelname == "WARNING" and "'file' command not found" in record.message for record in caplog.records
    )


def test_get_file_properties_file_command_error_logs_warning(tmp_path: Path, monkeypatch, caplog):
    import mimetypes

    test_file = tmp_path / "test.err"
    test_file.write_text("content")

    def mock_run_called_process_error(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "file", stderr="some error from file")

    monkeypatch.setattr(subprocess, "run", mock_run_called_process_error)
    monkeypatch.setattr(mimetypes, "guess_type", lambda path: ("application/octet-stream_fallback", None))
    caplog.clear()
    _, _, mime = get_file_properties(test_file)
    assert mime == "application/octet-stream_fallback"
    assert any(
        record.levelname == "WARNING"
        and "'file' command failed" in record.message
        and "some error from file" in record.message
        for record in caplog.records
    )
