import torch

from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.plugins.training_type import SingleDevicePlugin


def has_mps():
    mps_backend = getattr(torch.backends, "mps", None)
    return mps_backend is not None and mps_backend.is_available()


def get_torch_device(gpu_id=0, prefer_mps=True):
    if torch.cuda.is_available():
        return torch.device("cuda", gpu_id)
    if prefer_mps and has_mps():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_string(gpu_id=0, prefer_mps=True):
    return str(get_torch_device(gpu_id=gpu_id, prefer_mps=prefer_mps))


def get_module_device(module):
    for tensor in list(module.parameters()) + list(module.buffers()):
        return tensor.device
    return get_torch_device()


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def empty_device_cache(device=None):
    device = torch.device(device) if device is not None else get_torch_device()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


class SingleDeviceAccelerator(Accelerator):
    """Single-device accelerator that works with CUDA, MPS, or CPU in PL 1.5."""

    def setup(self, trainer):
        return super().setup(trainer)

    def get_device_stats(self, device):
        return {}

    @staticmethod
    def auto_device_count():
        return 1


def build_single_device_accelerator(device=None):
    device = torch.device(device) if device is not None else get_torch_device()
    return SingleDeviceAccelerator(
        precision_plugin=PrecisionPlugin(),
        training_type_plugin=SingleDevicePlugin(device=device),
    )
