"""H3 reference conditioning with one short, original-latent scene memory."""

from comfy_api.latest import io
from comfy_extras import nodes_minimax_h3 as h3

from .scene_memory import SceneMemoryError, _validate_video


class _MemoryClip:
    """Add one memory to the core encoder's own multimodal tokenization."""

    def __init__(self, clip, presentation):
        self.clip = clip
        self.presentation = presentation
        self.used = False

    def __getattr__(self, name):
        return getattr(self.clip, name)

    def tokenize(self, prompt, *args, **kwargs):
        if "minimax_ref_items" not in kwargs:
            raise SceneMemoryError("Unsupported H3 node: expected multimodal minimax_ref_items tokenization.")
        items = list(kwargs["minimax_ref_items"] or [])
        number = 1 + sum(item.get("type") == "video" for item in items)
        kwargs["minimax_ref_items"] = items + [self.presentation]
        note = (f"\n<Video {number}> is an earlier view of the scene. "
                "Use it as visual reference for the people, clothing, objects and setting. "
                "The new camera, action and dialogue follow the current description.")
        self.used = True
        return self.clip.tokenize(prompt + note, *args, **kwargs)

    def encode_from_tokens_scheduled(self, tokens, *args, **kwargs):
        conditioning = self.clip.encode_from_tokens_scheduled(tokens, *args, **kwargs)
        if not conditioning or any("minimax_token_tags" not in metadata for _, metadata in conditioning):
            raise SceneMemoryError("Scene memory requires an H3-compatible Qwen encoder with minimax_token_tags.")
        return conditioning


def _memory_window(scene_memory, vae, frame):
    if vae is None:
        raise SceneMemoryError("Scene memory requires the MiniMax H3 video VAE to present the window to Qwen.")
    if not isinstance(scene_memory, dict) or "samples" not in scene_memory:
        raise SceneMemoryError("Connect the video_latent output from Load Scene Memory.")
    video = _validate_video(scene_memory["samples"])
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise SceneMemoryError("memory_frame must be a non-negative source frame number.")
    total_frames = 5 + 17 * ((video.shape[2] - 2) // 5)
    if frame >= total_frames:
        raise SceneMemoryError(f"memory_frame must be less than the source's {total_frames} frames.")
    # The first two H3 latent steps represent five frames. Restarting this
    # temporal phase requires selecting a five-step (17-frame) boundary.
    start = (frame // 17) * 5
    window = video[:, :, start:start + 2, :, :].contiguous()
    if window.shape[2] != 2:
        raise SceneMemoryError("The selected H3 memory window is incomplete.")
    frames = vae.decode(window)
    if frames.ndim not in (4, 5):
        raise SceneMemoryError("The video VAE returned an unsupported frame tensor.")
    frames = frames.reshape(-1, *frames.shape[-3:])
    if (frames.shape[0] != 5 or frames.shape[1] != video.shape[3] * 16
            or frames.shape[2] != video.shape[4] * 16 or frames.shape[3] != 3):
        raise SceneMemoryError("The video VAE is incompatible with the saved H3 memory shape.")
    # A five-frame window sampled at H3's 2 fps yields its first frame;
    # the H3 tokenizer handles temporal padding of this single presentation.
    presentation = {"type": "video", "data": frames[:1].clone(), "timestamps": [0.0]}
    block = {"kind": "video", "latent_t": 2, "latent_h": window.shape[3],
             "latent_w": window.shape[4], "ref_audio_t": 0,
             "latent": window, "audio_latent": None}
    return presentation, block


class MiniMaxH3ReferenceToVideoSceneMemory(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        # Reuse the installed core schema, including every Autogrow ref input.
        schema = h3.MiniMaxH3ReferenceToVideo.define_schema()
        schema.node_id = "H3SceneMemoryReferenceToVideo"
        schema.display_name = "MiniMax H3 Reference to Video + Scene Memory"
        schema.category = "MiniMax H3/Scene Memory"
        schema.description = ("Native H3 reference conditioning with an optional original-latent scene memory. "
                              "When disabled the core node is called unchanged.")
        schema.inputs.extend([
            io.Boolean.Input("enabled", default=False, tooltip="Enable the scene-memory video reference."),
            io.Int.Input("memory_frame", default=0, min=0, max=3600,
                         tooltip="Source frame at 24 fps; window starts at the preceding 17-frame boundary."),
            io.Latent.Input("scene_memory", optional=True, lazy=True, tooltip="Video latent from Load Scene Memory."),
        ])
        return schema

    @classmethod
    def check_lazy_status(cls, enabled, scene_memory=None, **kwargs):
        return ["scene_memory"] if enabled and scene_memory is None else []

    @classmethod
    def execute(cls, clip, prompt, width, height, length, ref_image_size="match", vae=None, audio_vae=None,
                ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None,
                enabled=False, memory_frame=0, scene_memory=None):
        core_args = dict(clip=clip, prompt=prompt, width=width, height=height, length=length,
                         ref_image_size=ref_image_size, vae=vae, audio_vae=audio_vae,
                         ref_images=ref_images, ref_videos=ref_videos,
                         ref_video_audios=ref_video_audios, ref_audios=ref_audios)
        if not enabled:
            return h3.MiniMaxH3ReferenceToVideo.execute(**core_args)
        presentation, block = _memory_window(scene_memory, vae, memory_frame)
        wrapped = _MemoryClip(clip, presentation)
        core_args["clip"] = wrapped
        result = h3.MiniMaxH3ReferenceToVideo.execute(**core_args)
        if not wrapped.used:
            raise SceneMemoryError("Unsupported H3 node: the core did not tokenize the scene memory.")
        try:
            positive, latent = result.args
        except (AttributeError, ValueError, TypeError) as exc:
            raise SceneMemoryError("Unsupported H3 node output; update ComfyUI or SceneMemory.") from exc
        updated = []
        for embedding, meta in positive:
            if not isinstance(meta, dict):
                raise SceneMemoryError("Unsupported H3 conditioning metadata.")
            refs = list(meta.get("minimax_refs", []))
            updated.append([embedding, {**meta, "minimax_refs": refs + [block]}])
        return io.NodeOutput(updated, latent)
