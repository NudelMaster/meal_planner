"""Shared embedding device selection for app.py and ingest.py."""

import os
import subprocess
from typing import List, Optional, Tuple

# EmbeddingGemma-300m weights are ~600MB; leave headroom for runtime buffers.
DEFAULT_MIN_GPU_MEMORY_MB = 1536


def _min_gpu_memory_mb() -> int:
    raw = os.getenv("EMBED_MIN_GPU_MEMORY_MB", "").strip()
    if not raw:
        return DEFAULT_MIN_GPU_MEMORY_MB
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MIN_GPU_MEMORY_MB


def _gpu_free_mb_nvidia_smi() -> List[Tuple[int, int]]:
    """Return [(physical_gpu_index, free_mib), ...] without importing torch."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return []

    gpus: List[Tuple[int, int]] = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            gpus.append((int(parts[0]), int(parts[1])))
    return gpus


def _pick_most_free_gpu_smi(
    min_mb: int,
) -> Tuple[Optional[int], int, List[Tuple[int, int]]]:
    gpus = _gpu_free_mb_nvidia_smi()
    if not gpus:
        return None, 0, []

    qualifying = [(idx, free) for idx, free in gpus if free >= min_mb]
    if not qualifying:
        return None, 0, gpus

    best_idx, best_free = max(qualifying, key=lambda item: item[1])
    return best_idx, best_free, gpus


def configure_cuda_before_torch() -> None:
    """Set CUDA_VISIBLE_DEVICES before torch initializes CUDA.

    Must run before any ``import torch`` in the process. Uses nvidia-smi so GPU
    memory can be checked without touching cuda:0.
    """
    if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
        return

    forced = os.getenv("EMBED_DEVICE", "").strip().lower()
    if forced == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("EMBED_DEVICE=cpu: hiding CUDA devices")
        return

    min_mb = _min_gpu_memory_mb()
    gpus = _gpu_free_mb_nvidia_smi()
    if not gpus:
        return

    if forced.startswith("cuda") and ":" in forced:
        try:
            idx = int(forced.split(":", 1)[1])
        except ValueError:
            print(f"Invalid EMBED_DEVICE={forced!r}; auto-selecting GPU")
        else:
            free_by_idx = {gpu_idx: free for gpu_idx, free in gpus}
            if idx not in free_by_idx:
                print(f"EMBED_DEVICE={forced} is out of range; using CPU")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                return
            free_mb = free_by_idx[idx]
            if free_mb < min_mb:
                print(
                    f"EMBED_DEVICE={forced} has {free_mb}MB free "
                    f"(need {min_mb}MB); using CPU"
                )
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                return
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
            print(f"Set CUDA_VISIBLE_DEVICES={idx} ({free_mb}MB free)")
            return

    best_idx, best_free, all_gpus = _pick_most_free_gpu_smi(min_mb)
    if best_idx is None:
        free_summary = [free for _idx, free in all_gpus]
        print(
            f"No GPU with >={min_mb}MB free "
            f"(per-device free MB: {free_summary}); using CPU"
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)
    if len(all_gpus) > 1:
        free_summary = [free for _idx, free in all_gpus]
        print(
            f"Set CUDA_VISIBLE_DEVICES={best_idx} with most free memory "
            f"({best_free}MB; per-device free MB: {free_summary})"
        )
    else:
        print(f"Set CUDA_VISIBLE_DEVICES={best_idx} ({best_free}MB free)")


def _import_torch():
    import torch

    return torch


def _gpu_free_mb(device_index: int) -> int:
    torch = _import_torch()
    free, _total = torch.cuda.mem_get_info(device_index)
    return free // (1024 * 1024)


def _gpu_free_by_device() -> List[int]:
    torch = _import_torch()
    return [_gpu_free_mb(i) for i in range(torch.cuda.device_count())]


def _cuda_device_str(device_index: int) -> str:
    torch = _import_torch()
    if torch.cuda.device_count() == 1:
        return "cuda"
    return f"cuda:{device_index}"


def _pick_most_free_cuda() -> Tuple[Optional[str], int, List[int]]:
    """Return the visible CUDA device with the most free memory."""
    torch = _import_torch()
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return None, 0, []

    free_by_device = _gpu_free_by_device()
    best_idx = max(range(len(free_by_device)), key=free_by_device.__getitem__)
    return _cuda_device_str(best_idx), free_by_device[best_idx], free_by_device


def select_embed_device() -> str:
    """Pick CPU or the CUDA device with the most free memory.

    Honors optional env overrides:
    - EMBED_DEVICE=cpu | cuda | cuda:N — force device (falls back to CPU if invalid
      or under memory threshold)
    - EMBED_MIN_GPU_MEMORY_MB — minimum free VRAM required to use a GPU (default 1536)
    """
    torch = _import_torch()
    min_mb = _min_gpu_memory_mb()
    forced = os.getenv("EMBED_DEVICE", "").strip()
    if forced:
        lowered = forced.lower()
        if lowered == "cpu":
            return "cpu"
        if lowered.startswith("cuda"):
            if not torch.cuda.is_available():
                print("EMBED_DEVICE requests CUDA but it is unavailable; using CPU")
                return "cpu"
            if ":" in lowered:
                try:
                    idx = int(lowered.split(":", 1)[1])
                except ValueError:
                    print(f"Invalid EMBED_DEVICE={forced!r}; auto-selecting device")
                else:
                    if idx < 0 or idx >= torch.cuda.device_count():
                        print(f"EMBED_DEVICE={forced} is out of range; using CPU")
                        return "cpu"
                    free_mb = _gpu_free_mb(idx)
                    if free_mb < min_mb:
                        print(
                            f"EMBED_DEVICE={forced} has {free_mb}MB free "
                            f"(need {min_mb}MB); using CPU"
                        )
                        return "cpu"
                    return _cuda_device_str(idx)
            device, free_mb, free_by_device = _pick_most_free_cuda()
            if device is None:
                print("CUDA unavailable; using CPU")
                return "cpu"
            if free_mb < min_mb:
                print(
                    f"Best GPU has {free_mb}MB free (need {min_mb}MB; "
                    f"per-device free MB: {free_by_device}); using CPU"
                )
                return "cpu"
            if len(free_by_device) > 1:
                print(
                    f"Selected {device} with most free memory "
                    f"({free_mb}MB; per-device free MB: {free_by_device})"
                )
            return device
        print(f"Unrecognized EMBED_DEVICE={forced!r}; auto-selecting device")

    if not torch.cuda.is_available():
        print("CUDA unavailable; embedding on CPU")
        return "cpu"

    device, free_mb, free_by_device = _pick_most_free_cuda()
    if free_mb < min_mb:
        print(
            f"No GPU with >={min_mb}MB free "
            f"(per-device free MB: {free_by_device}); embedding on CPU"
        )
        return "cpu"

    if len(free_by_device) > 1:
        print(
            f"Selected {device} with most free memory "
            f"({free_mb}MB; per-device free MB: {free_by_device})"
        )
    else:
        print(f"Embedding on {device} ({free_mb}MB free)")
    return device


def is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error" in message
