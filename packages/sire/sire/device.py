import psutil
import torch


class DeviceManager:
    def __init__(self):
        self._devices = self._detect_devices()

    def _detect_devices(self) -> list[str]:
        if torch.cuda.is_available():
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        else:
            return ["cpu"]

    @property
    def devices(self) -> list[str]:
        return self._devices

    def get_available_memory(self, device: str) -> int:
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
            return torch.cuda.mem_get_info(device)[0]
        else:
            return psutil.virtual_memory().available

    def select_device(self, memory_required: int) -> str | None:
        for device in self.devices:
            if self.get_available_memory(device) > memory_required:
                return device
        return None
