"""ComfyUI nodes for stage-one scene memory storage."""

import os

import folder_paths

from .scene_memory import available_memories, file_fingerprint, load_memory, save_memory


def memory_folder():
    return os.path.join(folder_paths.get_output_directory(), "scene_memory")


class SaveSceneMemory:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video_latent": ("LATENT", {"tooltip": "Connect the video_latent output of Separate AV Latent after the Stage 1 H3 sampler."}),
            "filename": ("STRING", {"default": "scene_01", "tooltip": "Filename in output/scene_memory. Existing files are never overwritten."}),
        }}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("video_latent", "saved_path")
    FUNCTION = "save"
    CATEGORY = "MiniMax H3/Scene Memory"
    OUTPUT_NODE = True

    def save(self, video_latent, filename):
        path = save_memory(memory_folder(), filename, video_latent)
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
