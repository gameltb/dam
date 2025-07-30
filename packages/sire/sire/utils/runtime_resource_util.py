import ctypes

import accelerate
import psutil
import torch


def get_free_mem_size_cuda(device):
    mem_free_cuda, _ = torch.cuda.mem_get_info(device)
    return mem_free_cuda


def get_free_mem_size_cuda_pytorch(device):
    stats = torch.cuda.memory_stats(device)
    mem_active = stats["active_bytes.all.current"]
    mem_reserved = stats["reserved_bytes.all.current"]
    mem_free_cuda = get_free_mem_size_cuda(device)
    mem_free_torch = mem_reserved - mem_active
    mem_free_total = mem_free_cuda + mem_free_torch
    return mem_free_total


def get_free_mem_size_cpu():
    return psutil.virtual_memory().available


def libc_trim_memory():
    app_libc = ctypes.CDLL(None)
    return app_libc.malloc_trim(0)


def clear_device_cache_and_libc_alloc():
    accelerate.utils.memory.clear_device_cache(garbage_collection=True)
    libc_trim_memory()
