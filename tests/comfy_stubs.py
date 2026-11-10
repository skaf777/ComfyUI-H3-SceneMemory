"""Minimal Comfy doubles for node contract tests; not an H3 model."""
import sys
import types


class NodeOutput:
    def __init__(self, *args): self.args = args


class Input:
    def __init__(self, name, **kwargs): self.name, self.kwargs = name, kwargs


class InputType:
    Input = Input


class Schema:
    def __init__(self):
        self.node_id = "MiniMaxH3ReferenceToVideo"
        self.display_name = "MiniMax H3 Reference to Video"
        self.category = "original"
        self.description = "original"
        self.inputs = [Input(name) for name in ("clip", "vae", "audio_vae", "prompt", "width", "height", "length", "ref_image_size", "ref_images", "ref_videos", "ref_video_audios", "ref_audios")]
        self.outputs = ["positive", "latent"]


class Core:
    last_args = None
    last_result = None

    @classmethod
    def define_schema(cls): return Schema()

    @classmethod
    def execute(cls, **kwargs):
        cls.last_args = kwargs
        items = []
        refs = []
        for image in (kwargs.get("ref_images") or {}).values():
            if image is not None:
                items.append({"type": "image", "data": image})
                refs.append({"kind": "image", "value": image})
        audios = kwargs.get("ref_video_audios") or {}
        for name, video in (kwargs.get("ref_videos") or {}).items():
            if video is not None:
                audio = audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
                if audio is not None: items.append({"type": "audio"})
                items.append({"type": "video", "data": video})
                refs.append({"kind": "video_audio" if audio is not None else "video", "value": video, "audio": audio})
        for audio in (kwargs.get("ref_audios") or {}).values():
            if audio is not None:
                items.append({"type": "audio"})
                refs.append({"kind": "audio", "value": audio})
        clip = kwargs["clip"]
        tokens = clip.tokenize(kwargs["prompt"], minimax_ref_items=items)
        positive = clip.encode_from_tokens_scheduled(tokens)
        for _, meta in positive: meta["minimax_refs"] = refs
        cls.last_result = NodeOutput(positive, {"samples": "target AV latent"})
        return cls.last_result


def modules():
    io = types.SimpleNamespace(ComfyNode=object, Boolean=InputType, Int=InputType, Latent=InputType, NodeOutput=NodeOutput)
    api = types.ModuleType("comfy_api"); latest = types.ModuleType("comfy_api.latest"); latest.io = io
    extras = types.ModuleType("comfy_extras"); h3 = types.ModuleType("comfy_extras.nodes_minimax_h3")
    h3.MiniMaxH3ReferenceToVideo = Core; extras.nodes_minimax_h3 = h3
    return {"comfy_api": api, "comfy_api.latest": latest, "comfy_extras": extras,
            "comfy_extras.nodes_minimax_h3": h3}
