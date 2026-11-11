"""ComfyUI nodes for stage-one scene memory storage."""

import os
import logging
import tempfile

import numpy as np
import torch
from PIL import Image

import folder_paths

from .scene_memory import available_memories, file_fingerprint, load_memory, save_memory
from .reference_node import _memory_window

LOG = logging.getLogger("h3_scene_memory")


def save_thumbnail(path, latent, vae):
    intervals = (latent["samples"].shape[2] - 2) // 5
    # Two windows or fewer are very short; use the first in that case.
    # Otherwise choose the grid point nearest the video's midpoint.
    frame = 0 if intervals <= 1 else 17 * ((intervals + 1) // 2)
    presentation, _ = _memory_window(latent, vae, frame, validated=True)
    pixels = presentation["data"][0].detach().to(device="cpu", dtype=torch.float32).numpy()
    image = Image.fromarray((np.clip(pixels, 0, 1) * 255).astype(np.uint8), "RGB")
    image.thumbnail((256, 2560), Image.Resampling.LANCZOS)
    target = os.path.splitext(path)[0] + ".png"
    fd, temporary = tempfile.mkstemp(prefix=".h3scene-preview-", suffix=".tmp", dir=os.path.dirname(path))
    os.close(fd)
    try:
        image.save(temporary, format="PNG")
        os.link(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def memory_folder():
    return os.path.join(folder_paths.get_output_directory(), "scene_memory")


class SaveSceneMemory:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video_latent": ("LATENT", {"tooltip": "Connect the video_latent output of Separate AV Latent after the Stage 1 H3 sampler."}),
            "filename": ("STRING", {"default": "scene_01", "tooltip": "Filename in output/scene_memory. Existing files are never overwritten."}),
        }, "optional": {
            "auto_numbering": ("BOOLEAN", {"default": False, "tooltip": "Save each take under the next available number."}),
            "vae": ("VAE", {"tooltip": "Optional MiniMax H3 video VAE for a thumbnail."}),
        }}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("video_latent", "saved_path")
    FUNCTION = "save"
    CATEGORY = "MiniMax H3/Scene Memory"
    OUTPUT_NODE = True

    def save(self, video_latent, filename, auto_numbering=False, vae=None):
        path = save_memory(memory_folder(), filename, video_latent, auto_numbering=auto_numbering)
        if vae is not None:
            try:
                save_thumbnail(path, video_latent, vae)
            except Exception:
                LOG.exception("Could not create thumbnail for %s; the scene memory was saved", path)
        return video_latent, path


class LoadSceneMemory:
    @classmethod
    def INPUT_TYPES(cls):
        choices = available_memories(memory_folder()) or [""]
        return {"required": {"filename": (choices, {"tooltip": "Select a saved scene memory from output/scene_memory."})}}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("video_latent",)
    FUNCTION = "load"
    CATEGORY = "MiniMax H3/Scene Memory"

    @classmethod
    def IS_CHANGED(cls, filename):
        return file_fingerprint(memory_folder(), filename)

    def load(self, filename):
        return (load_memory(memory_folder(), filename),)
