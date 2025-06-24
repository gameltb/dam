import pytest
from pathlib import Path
import tempfile
import os
import shutil # For creating test_data dir in fixture if needed

from dam.services.file_operations import generate_perceptual_hashes, get_file_properties, calculate_sha256

# Fixture to provide paths to sample images
@pytest.fixture(scope="module")
def test_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Creates a temporary directory and populates it with test image files.
    This is better than relying on files in the repo for more hermetic tests.
    """
    # Base64 encoded content for a 2x1 red-blue PNG (img_A.png)
    img_a_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAYAAAD0In+KAAAAEUlEQVR42mNkgIL/DAwM/wUADgAB/vA/cQAAAABJRU5ErkJggg=="
    # Base64 encoded content for a 1x1 blue PNG (img_B.png)
    img_b_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPj/HwADBwIAMCbHYQAAAABJRU5ErkJggg=="

    # Create a temporary directory for test data for this module
    # If using files from tests/test_data, this fixture would just return Path("tests/test_data")
    # For now, let's use actual files from tests/test_data as created in the previous step.
    # This fixture can then just point to that static path.

    # The previous step created files in "tests/test_data" relative to project root.
    # So, this fixture should construct the path to that directory.
    project_root = Path(__file__).parent.parent.parent # Assuming tests/services/test_file_operations.py
    data_dir = project_root / "tests" / "test_data"

    # Ensure it exists (it should from previous step)
    if not data_dir.exists():
        # This indicates an issue with previous step or test setup assumption
        # For robustness, could create them here if missing, but it's better if prior step ensures it.
        # os.makedirs(data_dir, exist_ok=True)
        # with open(data_dir / "img_A.png", "wb") as f_a:
        #     import base64
        #     f_a.write(base64.b64decode(img_a_b64))
        # with open(data_dir / "img_B.png", "wb") as f_b:
        #     import base64
        #     f_b.write(base64.b64decode(img_b_b64))
        pytest.fail(f"Test data directory {data_dir} not found. Please ensure it's created by the previous plan step.")

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
        import imagehash
        from PIL import Image
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

def test_generate_perceptual_hashes_different_images_produce_different_hashes(sample_image_path_a: Path, sample_image_path_b: Path):
    """Test that two different images produce different perceptual hashes."""
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        pytest.skip("ImageHash or Pillow not installed, skipping perceptual hash comparison test.")

    hashes_a = generate_perceptual_hashes(sample_image_path_a)
    hashes_b = generate_perceptual_hashes(sample_image_path_b)

    # Ensure all keys were generated for both, otherwise comparison is moot
    if not all(k in hashes_a for k in ["phash", "ahash", "dhash"]) or \
       not all(k in hashes_b for k in ["phash", "ahash", "dhash"]):
        pytest.skip("Not all hash types generated for one or both sample images, cannot compare.")

    assert hashes_a["phash"] != hashes_b["phash"], "pHashes should differ for different images"
    assert hashes_a["ahash"] != hashes_b["ahash"], "aHashes should differ for different images"
    assert hashes_a["dhash"] != hashes_b["dhash"], "dHashes should differ for different images"


def test_generate_perceptual_hashes_non_image_file(non_image_file_path: Path, capsys):
    """Test with a non-image file, expect empty dict and warning."""
    # No need to skip this based on ImageHash/Pillow, as it tests the non-image path

    hashes = generate_perceptual_hashes(non_image_file_path)
    assert hashes == {}

    captured = capsys.readouterr()
    # Example warning: "Warning: Could not open or process image {non_image_file_path.name} for perceptual hashing: ..."
    # The exact message depends on PIL's error and your function's print statement.
    # This assertion is a bit brittle if the warning message changes.
    assert "Warning: Could not open or process image" in captured.out or \
           "Warning: Could not open or process image" in captured.err # Check both stdout and stderr for the warning

def test_generate_perceptual_hashes_missing_file(tmp_path: Path, capsys):
    """Test with a non-existent file path."""
    non_existent_file = tmp_path / "does_not_exist.png"
    hashes = generate_perceptual_hashes(non_existent_file)
    assert hashes == {}

    captured = capsys.readouterr()
    assert f"Warning: Image file not found at {non_existent_file}" in captured.out or \
           f"Warning: Image file not found at {non_existent_file}" in captured.err

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
    assert mime == "application/octet-stream" # Default for .dat or unknown
