"""ComfyUI-H3-SceneMemory custom nodes."""

from .nodes import LoadSceneMemory, SaveSceneMemory

NODE_CLASS_MAPPINGS = {
    "H3SceneMemorySave": SaveSceneMemory,
    "H3SceneMemoryLoad": LoadSceneMemory,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3SceneMemorySave": "Save Scene Memory",
    "H3SceneMemoryLoad": "Load Scene Memory",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
