"""H3 reference conditioning with phase-aligned original-latent scene memories."""

import logging
import torch
from comfy_api.latest import io
from comfy_extras import nodes_minimax_h3 as h3

from .scene_memory import SceneMemoryError, _digest, _validate_video
from .selection import MODES, PRESETS, select

LOG = logging.getLogger("h3_scene_memory")
MEMORY_INPUTS = ("scene_memory", "scene_memory_2", "scene_memory_3", "scene_memory_4")


class _MemoryClip:
    def __init__(self, clip, presentations):
        self.clip, self.presentations, self.used = clip, presentations, False

    def __getattr__(self, name):
        return getattr(self.clip, name)

    def tokenize(self, prompt, *args, **kwargs):
        if "minimax_ref_items" not in kwargs:
            raise SceneMemoryError("Unsupported H3 node: expected multimodal minimax_ref_items tokenization.")
        items = list(kwargs["minimax_ref_items"] or [])
        first = 1 + sum(item.get("type") == "video" for item in items)
        kwargs["minimax_ref_items"] = items + self.presentations
        labels = ", ".join(f"<Video {i}>" for i in range(first, first + len(self.presentations)))
        note = (f"\n{labels} are earlier views of the scene. "
                "Use them as visual references for the people, clothing, objects and setting. "
                "The new camera, action and dialogue follow the current description.")
        self.used = True
        return self.clip.tokenize(prompt + note, *args, **kwargs)

    def encode_from_tokens_scheduled(self, tokens, *args, **kwargs):
        conditioning = self.clip.encode_from_tokens_scheduled(tokens, *args, **kwargs)
        if not conditioning or any("minimax_token_tags" not in metadata for _, metadata in conditioning):
            raise SceneMemoryError("Scene memory requires an H3-compatible Qwen encoder with minimax_token_tags.")
        return conditioning


def _memory_window(scene_memory, vae, frame, validated=False):
    if vae is None:
        raise SceneMemoryError("Scene memory requires the MiniMax H3 video VAE to present windows to Qwen.")
    if not isinstance(scene_memory, dict) or "samples" not in scene_memory:
        raise SceneMemoryError("Connect the video_latent output from Load Scene Memory.")
    video = scene_memory["samples"] if validated else _validate_video(scene_memory["samples"])
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise SceneMemoryError("memory_frame must be a non-negative source frame number.")
    total_frames = 5 + 17 * ((video.shape[2] - 2) // 5)
    if frame >= total_frames:
        raise SceneMemoryError(f"memory_frame must be less than the source's {total_frames} frames.")
    start = (frame // 17) * 5
    window = video[:, :, start:start + 2, :, :].contiguous().clone()
    if window.shape[2] != 2:
        raise SceneMemoryError("The selected H3 memory window is incomplete.")
    # Keep both the saved tensor and the DiT payload isolated from the VAE.
    frames = vae.decode(window.clone())
    if frames.ndim not in (4, 5):
        raise SceneMemoryError("The video VAE returned an unsupported frame tensor.")
    frames = frames.reshape(-1, *frames.shape[-3:])
    if (frames.shape[0] != 5 or frames.shape[1] != video.shape[3] * 16
            or frames.shape[2] != video.shape[4] * 16 or frames.shape[3] != 3):
        raise SceneMemoryError("The video VAE is incompatible with the saved H3 memory shape.")
    presentation = {"type": "video", "data": frames[:1].clone(), "timestamps": [0.0]}
    block = {"kind": "video", "latent_t": 2, "latent_h": window.shape[3],
             "latent_w": window.shape[4], "ref_audio_t": 0,
             "latent": window, "audio_latent": None}
    return presentation, block


class MiniMaxH3ReferenceToVideoSceneMemory(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema = h3.MiniMaxH3ReferenceToVideo.define_schema()
        schema.node_id = "H3SceneMemoryReferenceToVideo"
        schema.display_name = "MiniMax H3 Reference to Video + Scene Memory"
        schema.category = "MiniMax H3/Scene Memory"
        schema.description = "Native H3 references with optional short original-latent scene-memory windows."
        # Keep the original three fields in their old order for saved workflows.
        schema.inputs.extend([
            io.Boolean.Input("enabled", default=False, tooltip="Enable scene memory."),
            io.Int.Input("memory_frame", default=0, min=0, max=3600, tooltip="Single mode: source frame, snapped down to the 17-frame grid."),
            io.Latent.Input("scene_memory", optional=True, lazy=True, tooltip="First Load Scene Memory output."),
            io.Combo.Input("memory_mode", options=list(MODES), default="Uniform", optional=True),
            io.Combo.Input("memory_preset", options=[*PRESETS, "Custom"], default="Balanced", optional=True,
                           tooltip="Selecting a preset changes Max References; you can edit it afterwards."),
            io.Int.Input("max_references", default=3, min=1, max=2147483647, optional=True),
            io.Int.Input("frame_interval", default=17, min=1, max=3600, optional=True, tooltip="Interval mode, in source frames; e.g. 5, 10, 17, 34, 68."),
            io.String.Input("manual_frames", default="", optional=True, tooltip="Comma-separated frame numbers; for multiple files use 1:0, 2:187."),
            io.String.Input("pinned_frames", default="", optional=True, tooltip="Priority frames, e.g. 0, 187 or 1:0, 2:187."),
            *[io.Latent.Input(name, optional=True, lazy=True, tooltip="Optional additional Load Scene Memory output, explicitly chosen for this scene.")
              for name in MEMORY_INPUTS[1:]],
        ])
        return schema

    @classmethod
    def check_lazy_status(cls, enabled, **kwargs):
        if not enabled:
            return []
        # Only connected lazy inputs appear in kwargs with None. Do not request
        # absent optional sockets: ComfyUI cannot make a missing link strong.
        return [name for name in MEMORY_INPUTS if name in kwargs and kwargs[name] is None]

    @classmethod
    def execute(cls, clip, prompt, width, height, length, ref_image_size="match", vae=None, audio_vae=None,
                ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None,
                enabled=False, memory_frame=0, scene_memory=None, memory_mode="Single", memory_preset="Custom",
                max_references=3, frame_interval=17, manual_frames="", pinned_frames="",
                scene_memory_2=None, scene_memory_3=None, scene_memory_4=None):
        core_args = dict(clip=clip, prompt=prompt, width=width, height=height, length=length,
                         ref_image_size=ref_image_size, vae=vae, audio_vae=audio_vae,
                         ref_images=ref_images, ref_videos=ref_videos,
                         ref_video_audios=ref_video_audios, ref_audios=ref_audios)
        if not enabled:
            return h3.MiniMaxH3ReferenceToVideo.execute(**core_args)
        if vae is None:
            raise SceneMemoryError("Scene memory requires the MiniMax H3 video VAE.")
        if scene_memory is None:
            raise SceneMemoryError("Connect the first Load Scene Memory output to scene_memory.")
        if memory_preset not in (*PRESETS, "Custom"):
            raise SceneMemoryError("Unknown memory preset.")
        sources = []
        for slot, memory in enumerate((scene_memory, scene_memory_2, scene_memory_3, scene_memory_4), 1):
            if memory is None:
                continue
            if not isinstance(memory, dict) or "samples" not in memory:
                raise SceneMemoryError(f"Memory file {slot} is not a video LATENT.")
            video = _validate_video(memory["samples"])
            cpu = video.detach().to(device="cpu").contiguous()
            identity = (tuple(video.shape), str(video.dtype), _digest(cpu))
            sources.append((slot, video, identity))
        by_slot = {slot: video for slot, video, _ in sources}
        cache = {}
        def get_window(key):
            if key not in cache:
                cache[key] = _memory_window({"samples": by_slot[key[0]]}, vae, key[1], validated=True)
            return cache[key]
        def render(key):
            frame = get_window(key)[0]["data"][0]
            return frame.detach().to(device="cpu", dtype=torch.float32).numpy()
        selected, warnings = select(sources, memory_mode, max_references, memory_frame, frame_interval,
                                    manual_frames, pinned_frames, render)
        if not selected:
            raise SceneMemoryError("No scene-memory windows were selected.")
        # This counts DiT video patch rows only; Qwen and existing refs cost extra.
        rows = sum(2 * (by_slot[slot].shape[3] // 2) * (by_slot[slot].shape[4] // 2)
                   for slot, _ in selected)
        if rows >= 8192 or len(selected) >= 8:
            warnings.append(f"Scene memory adds approximately {rows} DiT visual tokens across {len(selected)} windows, plus Qwen, target and existing references; VRAM, context length and attention cost may be high. This is a heuristic, not an H3 limit.")
        report = "Selected scene memory: " + ", ".join(f"{slot}:{frame}" for slot, frame in selected)
        LOG.info(report)
        for warning in warnings:
            LOG.warning("Scene memory: %s", warning)
        windows = [get_window(key) for key in selected]
        wrapped = _MemoryClip(clip, [presentation for presentation, _ in windows])
        core_args["clip"] = wrapped
        result = h3.MiniMaxH3ReferenceToVideo.execute(**core_args)
        if not wrapped.used:
            raise SceneMemoryError("Unsupported H3 node: the core did not tokenize the scene memory.")
        try:
            positive, latent = result.args
        except (AttributeError, ValueError, TypeError) as exc:
            raise SceneMemoryError("Unsupported H3 node output; update ComfyUI or SceneMemory.") from exc
        blocks = [block for _, block in windows]
        # Existing native reference rows are additive to memory rows. Count
        # known stock descriptors; text and target sequence also consume budget.
        if not positive or not isinstance(positive[0][1], dict):
            raise SceneMemoryError("Unsupported H3 conditioning metadata.")
        first_meta = positive[0][1]
        existing_rows = 0
        for ref in first_meta.get("minimax_refs", []):
            kind = ref.get("kind")
            if kind in ("image", "video", "video_audio") and "latent_h" in ref and "latent_w" in ref:
                existing_rows += int(ref.get("latent_t", 1)) * (int(ref["latent_h"]) // 2) * (int(ref["latent_w"]) // 2)
            if kind in ("audio", "video_audio"):
                existing_rows += 2 * int(ref.get("ref_audio_t", 0))
        if existing_rows + rows >= 16384:
            warning = f"Known H3 reference blocks add approximately {existing_rows + rows} DiT tokens before text and target AV; context and attention may be costly."
            warnings.append(warning)
            LOG.warning("Scene memory: %s", warning)
        updated = []
        for embedding, meta in positive:
            if not isinstance(meta, dict):
                raise SceneMemoryError("Unsupported H3 conditioning metadata.")
            updated.append([embedding, {**meta, "minimax_refs": list(meta.get("minimax_refs", [])) + blocks}])
        return io.NodeOutput(updated, latent, ui={"text": [report, *warnings]})
