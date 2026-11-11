"""Storage for original MiniMax H3 stage-one video latents."""

import hashlib
import json
import os
import re
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

FORMAT = "h3_scene_memory_v1"
TENSOR_NAME = "video_latent"
SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


class SceneMemoryError(ValueError):
    """A memory file or its latent is invalid or incompatible."""


def _filename(name):
    if not isinstance(name, str):
        raise SceneMemoryError("Memory filename must be text.")
    name = name.strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise SceneMemoryError("Enter a filename without a path.")
    if any(ord(c) < 32 or c in '<>:"|?*' for c in name) or name.endswith((".", " ")):
        raise SceneMemoryError("Invalid memory filename.")
    if not name.lower().endswith(".safetensors"):
        name += ".safetensors"
    return name


def _path(folder, name):
    folder = os.path.realpath(folder)
    path = os.path.join(folder, _filename(name))
    if os.path.dirname(os.path.realpath(path)) != folder:
        raise SceneMemoryError("Memory file must stay inside the scene_memory folder.")
    return path


def _validate_video(video):
    if not isinstance(video, torch.Tensor) or video.is_nested:
        raise SceneMemoryError("Expected a video-only H3 tensor, not an AV NestedTensor. Use Separate AV Latent first.")
    if video.ndim != 5:
        raise SceneMemoryError("Expected video latent shape [1, 24, T, H, W].")
    batch, channels, t, height, width = video.shape
    if batch != 1 or channels != 24 or t < 2 or (t - 2) % 5:
        raise SceneMemoryError("Invalid H3 video shape or temporal grid (expected batch 1, 24 channels and T = 5k+2).")
    if height < 2 or width < 2 or height % 2 or width % 2:
        raise SceneMemoryError("H3 latent height and width must be positive even values.")
    if video.dtype not in SUPPORTED_DTYPES:
        raise SceneMemoryError("H3 video latent dtype must be float16, bfloat16 or float32.")
    if not bool(torch.isfinite(video).all().item()):
        raise SceneMemoryError("Video latent contains NaN or infinity.")
    return video


def _digest(video):
    # The byte view preserves all bits, including bfloat16; never convert dtype.
    data = video.contiguous().view(torch.uint8).reshape(-1).numpy()
    return hashlib.sha256(memoryview(data)).hexdigest()


def _metadata(video):
    _, _, t, h, w = video.shape
    return {
        "format": FORMAT,
        "source": "stage1",
        "shape": json.dumps(list(video.shape), separators=(",", ":")),
        "dtype": str(video.dtype).removeprefix("torch."),
        "width": str(w * 16),
        "height": str(h * 16),
        "latent_frames": str(t),
        "frame_count": str(5 + 17 * ((t - 2) // 5)),
        "fps": "24",
        "sha256": _digest(video),
    }


def save_memory(folder, filename, latent, auto_numbering=False):
    if not isinstance(latent, dict) or "samples" not in latent:
        raise SceneMemoryError("Expected a ComfyUI LATENT with samples.")
    video = _validate_video(latent["samples"])
    # Copy to CPU without changing precision or sharing mutable input storage.
    video = video.detach().to(device="cpu", copy=True).contiguous()
    meta = _metadata(video)
    os.makedirs(folder, exist_ok=True)
    path = _path(folder, filename)
    stem = os.path.basename(path)[:-len('.safetensors')]
    fd, temporary = tempfile.mkstemp(prefix=".h3scene-", suffix=".tmp", dir=folder)
    os.close(fd)
    try:
        save_file({TENSOR_NAME: video}, temporary, metadata=meta)
        # Atomic publication without replacing an existing scene memory.
        while True:
            if auto_numbering:
                pattern = re.compile(r'^' + re.escape(stem) + r'_(\d{4,})\.safetensors$', re.IGNORECASE)
                numbers = [int(match.group(1)) for name in os.listdir(folder)
                           if (match := pattern.fullmatch(name))]
                path = _path(folder, f"{stem}_{max(numbers, default=0) + 1:04d}.safetensors")
            try:
                os.link(temporary, path)
                break
            except FileExistsError as exc:
                if not auto_numbering:
                    raise SceneMemoryError(f"Memory file already exists: {os.path.basename(path)}") from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_memory(folder, filename):
    path = _path(folder, filename)
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            if list(f.keys()) != [TENSOR_NAME]:
                raise SceneMemoryError("Memory file must contain exactly one video_latent tensor.")
        tensors = load_file(path, device="cpu")
    except SceneMemoryError:
        raise
    except Exception as exc:
        raise SceneMemoryError(f"Cannot read scene memory '{os.path.basename(path)}': {exc}") from exc
    video = _validate_video(tensors[TENSOR_NAME])
    expected = _metadata(video)
    for key, value in expected.items():
        if meta.get(key) != value:
            raise SceneMemoryError(f"Scene memory metadata or checksum mismatch: {key}.")
    return {"samples": video}


def available_memories(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(name for name in os.listdir(folder)
                  if name.lower().endswith(".safetensors") and os.path.isfile(os.path.join(folder, name))
                  and os.path.dirname(os.path.realpath(os.path.join(folder, name))) == os.path.realpath(folder))


def file_fingerprint(folder, filename):
    try:
        path = _path(folder, filename)
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except (OSError, SceneMemoryError):
        return float("nan")
