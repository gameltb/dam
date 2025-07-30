from unittest.mock import patch

from sire.device import DeviceManager


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.device_count", return_value=2)
def test_detect_devices_gpu(mock_device_count, mock_is_available):
    manager = DeviceManager()
    assert manager.devices == ["cuda:0", "cuda:1"]


@patch("torch.cuda.is_available", return_value=False)
def test_detect_devices_cpu(mock_is_available):
    manager = DeviceManager()
    assert manager.devices == ["cpu"]


@patch("sire.device.DeviceManager.get_available_memory")
def test_select_device_with_enough_memory(mock_get_available_memory):
    mock_get_available_memory.side_effect = [100, 500]  # cuda:0 has 100, cuda:1 has 500

    with patch.object(DeviceManager, "_detect_devices", return_value=["cuda:0", "cuda:1"]):
        manager = DeviceManager()
        selected_device = manager.select_device(memory_required=200)
        assert selected_device == "cuda:1"


@patch("sire.device.DeviceManager.get_available_memory")
def test_select_device_no_device_with_enough_memory(mock_get_available_memory):
    mock_get_available_memory.side_effect = [100, 150]  # cuda:0 has 100, cuda:1 has 150

    with patch.object(DeviceManager, "_detect_devices", return_value=["cuda:0", "cuda:1"]):
        manager = DeviceManager()
        selected_device = manager.select_device(memory_required=200)
        assert selected_device is None
