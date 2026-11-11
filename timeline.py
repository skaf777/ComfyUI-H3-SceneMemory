"""Independent preview and persisted selection for H3 memory windows."""
import base64
import io
import json
import weakref
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

from .reference_node import _memory_window
from .scene_memory import SceneMemoryError, _digest, _validate_video
from .selection import candidates

_CACHE = OrderedDict()
_CACHE_LIMIT = 512
_VAE_IDS = weakref.WeakKeyDictionary()
_NEXT_VAE_ID = 0


def source_id(video):
    cpu = video.detach().to(device="cpu").contiguous()
    return _digest(cpu)


def sources_from_memories(memories):
    sources = []
    for slot, memory in enumerate(memories, 1):
        if memory is None:
            continue
        if not isinstance(memory, dict) or "samples" not in memory:
            raise SceneMemoryError(f"Memory file {slot} is not a video LATENT.")
        video = _validate_video(memory["samples"])
        sources.append((slot, video, source_id(video)))
    return sources


def selection_for_sources(selection, sources):
    if not isinstance(selection, dict) or selection.get("version") != 1:
        raise SceneMemoryError("Invalid Timeline selection. Refresh previews and select frames again.")
    ids = {slot: identity for slot, _, identity in sources}
    available = set(candidates(sources))
    result, seen = [], set()
    for item in selection.get("items", []):
        if not isinstance(item, dict):
            raise SceneMemoryError("Invalid Timeline selection item.")
        slot, frame, identity = item.get("slot"), item.get("frame"), item.get("source")
        if type(slot) is not int or type(frame) is not int or ids.get(slot) != identity:
            raise SceneMemoryError(f"Timeline source mismatch in memory slot {slot}; reconnect the matching file and refresh previews.")
        if (slot, frame) not in available:
            raise SceneMemoryError(f"Timeline frame {frame} is no longer available in memory slot {slot}.")
        key = (identity, frame)
        if key not in seen:
            seen.add(key)
            result.append((slot, frame))
    if not result:
        raise SceneMemoryError("Timeline selection is empty. Select at least one preview.")
    return result


def preview(video, identity, vae, frame):
    global _NEXT_VAE_ID
    if vae not in _VAE_IDS:
        _NEXT_VAE_ID += 1
        _VAE_IDS[vae] = _NEXT_VAE_ID
    key = (identity, _VAE_IDS[vae], frame)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    presentation, _ = _memory_window({"samples": video}, vae, frame, validated=True)
    image = presentation["data"][0].detach().to(device="cpu", dtype=torch.float32).numpy()
    image = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8), "RGB")
    image.thumbnail((128, 1280), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    _CACHE[key] = encoded
    if len(_CACHE) > _CACHE_LIMIT:
        _CACHE.popitem(last=False)
    return encoded


class SceneMemoryTimeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "scene_memory": ("LATENT", {"lazy": True}), "vae": ("VAE", {"lazy": True}),
            "preview_interval": (["17", "34", "51", "68", "Custom"], {"default": "17"}),
            "custom_interval": ("INT", {"default": 17, "min": 1, "max": 2147483647}),
            "selection": ("STRING", {"default": '{"version":1,"items":[]}', "multiline": False}),
        }, "optional": {**{name: ("LATENT", {"lazy": True}) for name in ("scene_memory_2", "scene_memory_3", "scene_memory_4")},
                        "preview_request": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("USER_SELECTION",)
    RETURN_NAMES = ("USER_SELECTION",)
    FUNCTION = "execute"
    CATEGORY = "MiniMax H3/Scene Memory"
    OUTPUT_NODE = True

    def check_lazy_status(self, preview_request="", **kwargs):
        if not preview_request:
            return []
        names = ("scene_memory", "vae", "scene_memory_2", "scene_memory_3", "scene_memory_4")
        return [name for name in names if name in kwargs and kwargs[name] is None]

    def execute(self, scene_memory=None, vae=None, preview_interval="17", custom_interval=17, selection='{"version":1,"items":[]}',
                scene_memory_2=None, scene_memory_3=None, scene_memory_4=None, preview_request=""):

        try:
            saved = json.loads(selection)
        except (ValueError, TypeError) as exc:
            raise SceneMemoryError("Timeline selection is not valid JSON.") from exc
        if not isinstance(saved, dict) or saved.get("version") != 1 or not isinstance(saved.get("items"), list):
            raise SceneMemoryError("Timeline selection is invalid.")
        if not preview_request:
            return (saved,)
        sources = sources_from_memories((scene_memory, scene_memory_2, scene_memory_3, scene_memory_4))
        interval = custom_interval if preview_interval == "Custom" else int(preview_interval)
        if type(interval) is not int or interval < 1:
            raise SceneMemoryError("Preview interval must be positive.")
        # Sample the H3 grid once; intervals below 17 cannot add unique windows.
        step = max(17, ((interval + 16) // 17) * 17)
        images = []
        for slot, video, identity in sources:
            for frame in range(0, 17 * ((video.shape[2] - 2) // 5) + 1, step):
                images.append({"slot": slot, "frame": frame, "source": identity,
                               "png": preview(video, identity, vae, frame)})
        ids = {slot: identity for slot, _, identity in sources}
        available = set(candidates(sources))
        stale = any(not isinstance(item, dict) or ids.get(item.get("slot")) != item.get("source")
                    or (item.get("slot"), item.get("frame")) not in available for item in saved["items"])
        return {"ui": {"previews": images, "warning": ["Saved selection does not match the connected source. Clear it and select again."] if stale else []},
                "result": (saved,)}
