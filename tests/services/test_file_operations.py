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
    # The base64 variables img_a_b64 and img_b_b64 were for creating files on the fly.
    # Now that the files are static in tests/test_data (created by plan step),
    # these variables are unused in this fixture.

    # Create a temporary directory for test data for this module
    # If using files from tests/test_data, this fixture would just return Path("tests/test_data")
    data_dir = tmp_path_factory.mktemp("file_ops_test_data")

    # Helper to create dummy image files locally for this module's tests
    def _create_dummy_image_for_file_ops(filepath: Path, color_name: str, size=(10, 10)):
        colors = {
            "red": (255, 0, 0),
            "blue": (0, 0, 255),
        }
        try:
            from PIL import Image

            img = Image.new("RGB", size, color=colors[color_name])
            if color_name == "blue":  # Make it slightly different from red for comparison tests
                for i in range(size[0]):
                    img.putpixel((i, i), (255, 255, 255))  # White diagonal line
            filepath.parent.mkdir(parents=True, exist_ok=True)
            img.save(filepath, "PNG")
        except ImportError:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(f"dummy_image_{color_name}".encode())

    _create_dummy_image_for_file_ops(data_dir / "img_A.png", "red")
    _create_dummy_image_for_file_ops(data_dir / "img_B.png", "blue")
    # Add any other files needed by tests in this module

    return data_dir


@pytest.fixture
def sample_image_path_a(test_data_dir: Path) -> Path:
    return test_data_dir / "img_A.png"


@pytest.fixture
def sample_image_path_b(test_data_dir: Path) -> Path:
    return test_data_dir / "img_B.png"


@pytest.fixture
def non_image_file_path(tmp_path: Path) -> Path:
    """Creates a temporary non-image file (text file)."""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("This is not an image.")
    return txt_file


def test_generate_perceptual_hashes_valid_image(sample_image_path_a: Path):
    """Test with a valid image, check for expected hash types."""
    # Ensure ImageHash and Pillow are available for this test to run meaningfully
    try:
        import imagehash  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        pytest.skip("ImageHash or Pillow not installed, skipping perceptual hash generation test.")

    hashes = generate_perceptual_hashes(sample_image_path_a)

    assert isinstance(hashes, dict)
    possible_keys = {"phash", "ahash", "dhash"}

    # Check that at least one expected hash was generated
    assert any(key in hashes for key in possible_keys), "No perceptual hashes were generated."

    # For any hashes that were generated, check they are valid strings
    for key in possible_keys:
        if key in hashes:
            assert isinstance(hashes[key], str), f"{key} value is not a string."
            assert len(hashes[key]) > 0, f"{key} value is an empty string."

    # Example: Known hash for a very specific image if available (hard to maintain)
    # assert hashes["phash"] == "expected_phash_value_for_img_a"


def test_generate_perceptual_hashes_different_images_produce_different_hashes(
    sample_image_path_a: Path, sample_image_path_b: Path
):
    """Test that two different images produce different perceptual hashes."""
    try:
        import imagehash  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        pytest.skip("ImageHash or Pillow not installed, skipping perceptual hash comparison test.")

    hashes_a = generate_perceptual_hashes(sample_image_path_a)
    hashes_b = generate_perceptual_hashes(sample_image_path_b)

    # Ensure all keys were generated for both, otherwise comparison is moot
    if not all(k in hashes_a for k in ["phash", "ahash", "dhash"]) or not all(
        k in hashes_b for k in ["phash", "ahash", "dhash"]
    ):
        pytest.skip("Not all hash types generated for one or both sample images, cannot compare.")

    assert hashes_a["phash"] != hashes_b["phash"], "pHashes should differ for different images"
    assert hashes_a["ahash"] != hashes_b["ahash"], "aHashes should differ for different images"
    assert hashes_a["dhash"] != hashes_b["dhash"], "dHashes should differ for different images"


def test_generate_perceptual_hashes_non_image_file(non_image_file_path: Path, caplog):  # This is the good one
    """Test with a non-image file, expect empty dict and warning."""
    caplog.clear()
    hashes = generate_perceptual_hashes(non_image_file_path)
    assert hashes == {}

    assert len(caplog.records) > 0
    found_log = False
    for record in caplog.records:
        # Check for part of the expected log message
        if (
            record.levelname == "WARNING"
            and "Could not open or process image" in record.message
            and non_image_file_path.name in record.message
        ):
            found_log = True
            break
    assert found_log, "Expected warning for non-image file not found in logs."


def test_generate_perceptual_hashes_missing_file(tmp_path: Path, caplog):
    """Test with a non-existent file path."""
    non_existent_file = tmp_path / "does_not_exist.png"

    # Clear previous log captures if any from other tests or setups
    caplog.clear()

    hashes = generate_perceptual_hashes(non_existent_file)
    assert hashes == {}

    assert len(caplog.records) > 0
    found_log = False
    for record in caplog.records:
        if record.levelname == "WARNING" and f"Image file not found at {non_existent_file}" in record.message:
            found_log = True
            break
    assert found_log, "Expected warning for missing file not found in logs."


# Test for conditional import (harder to test directly without manipulating sys.modules or environments)
# One way is to check the _imagehash_available flags if they were made accessible for testing,
# or by mocking the import statement.
# For now, the function's internal check `if not _imagehash_available...` covers this.
# A test could be added if we make those flags importable or have a helper.


# Example test for calculate_sha256 (already exists but good to have tests for all ops)
def test_calculate_sha256_simple_file(tmp_path: Path):
    file_content = b"hello world"
    test_file = tmp_path / "sha_test.txt"
    test_file.write_bytes(file_content)

    # Expected SHA256 for "hello world"
    # echo -n "hello world" | sha256sum
    # b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
    expected_hash = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert calculate_sha256(test_file) == expected_hash


# Example test for get_file_properties
def test_get_file_properties_simple_file(tmp_path: Path):
    file_content = b"some data"
    test_file = tmp_path / "props_test.dat"
    test_file.write_bytes(file_content)

    name, size, mime = get_file_properties(test_file)
    assert name == "props_test.dat"
    assert size == len(file_content)
    # This will depend on whether 'file' command is available and its output
    # For a .dat file, 'file' might also return application/octet-stream or something else
    # If 'file' command is not present, it will fallback to mimetypes, then to application/octet-stream
    # To make this test deterministic, we should mock subprocess.run
    assert mime is not None # Check that a mime type is returned


# Tests for MIME type detection in get_file_properties
import subprocess # Import subprocess here

@pytest.mark.parametrize(
    "mock_file_output, mock_mimetypes_guess, expected_mime, file_ext",
    [
        # Scenario 1: 'file' command succeeds
        ("image/png", None, "image/png", ".png"),
        ("text/plain", None, "text/plain", ".txt"),
        # Scenario 2: 'file' command fails (e.g., FileNotFoundError), fallback to mimetypes
        (FileNotFoundError(), "image/jpeg", "image/jpeg", ".jpg"),
        (FileNotFoundError(), None, "application/octet-stream", ".unknown"), # mimetypes also fails
        # Scenario 3: 'file' command throws CalledProcessError, fallback to mimetypes
        (subprocess.CalledProcessError(1, "file cmd"), "application/pdf", "application/pdf", ".pdf"),
        # Scenario 4: 'file' command available but returns nothing useful (empty string), fallback to mimetypes
        ("", "text/xml", "text/xml", ".xml"),
         # Scenario 5: Both 'file' and mimetypes fail
        (FileNotFoundError(), None, "application/octet-stream", ".dat"),
        ("", None, "application/octet-stream", ".foo"),
    ],
)
def test_get_file_properties_mime_detection_logic(
    tmp_path: Path,
    monkeypatch,
    mock_file_output,
    mock_mimetypes_guess,
    expected_mime,
    file_ext
):
    import subprocess # Import here for the CalledProcessError type
    import mimetypes   # For monkeypatching

    test_file = tmp_path / f"testfile{file_ext}"
    test_file.write_text("dummy content")

    mock_subprocess_run = None
    if isinstance(mock_file_output, Exception):
        def _mock_subprocess_run_exception(*args, **kwargs):
            raise mock_file_output
        mock_subprocess_run = _mock_subprocess_run_exception
    else: # String output
        def _mock_subprocess_run_success(*args, **kwargs):
            # Simulate successful run of 'file' command
            # args[0] should be ['file', '-b', '--mime-type', str(test_file)]
            return subprocess.CompletedProcess(args[0], 0, stdout=mock_file_output, stderr="")
        mock_subprocess_run = _mock_subprocess_run_success

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)
    monkeypatch.setattr(mimetypes, "guess_type", lambda path: (mock_mimetypes_guess, None))

    _, _, mime_type = get_file_properties(test_file)
    assert mime_type == expected_mime


def test_get_file_properties_file_command_not_present_logs_warning(tmp_path: Path, monkeypatch, caplog):
    """Ensure a warning is logged if 'file' command is not found."""
    import mimetypes
    import subprocess

    test_file = tmp_path / "test.txt"
    test_file.write_text("content")

    def mock_run_file_not_found(*args, **kwargs):
        raise FileNotFoundError("'file' command not found")

    monkeypatch.setattr(subprocess, "run", mock_run_file_not_found)
    # Let mimetypes succeed
    monkeypatch.setattr(mimetypes, "guess_type", lambda path: ("text/plain_fallback", None))
    caplog.clear()

    _, _, mime = get_file_properties(test_file)
    assert mime == "text/plain_fallback"
    assert any(
        record.levelname == "WARNING" and "'file' command not found" in record.message
        for record in caplog.records
    ), "Warning for 'file' command not found was not logged."


def test_get_file_properties_file_command_error_logs_warning(tmp_path: Path, monkeypatch, caplog):
    """Ensure a warning is logged if 'file' command fails with CalledProcessError."""
    import mimetypes
    import subprocess

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
        record.levelname == "WARNING" and "'file' command failed" in record.message and "some error from file" in record.message
        for record in caplog.records
    ), "Warning for 'file' command failure was not logged correctly."
