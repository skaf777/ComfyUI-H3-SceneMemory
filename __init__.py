"""ComfyUI-H3-SceneMemory custom nodes."""

from .nodes import LoadSceneMemory, SaveSceneMemory
from .reference_node import MiniMaxH3ReferenceToVideoSceneMemory

NODE_CLASS_MAPPINGS = {
    "H3SceneMemorySave": SaveSceneMemory,
    "H3SceneMemoryLoad": LoadSceneMemory,
    "H3SceneMemoryReferenceToVideo": MiniMaxH3ReferenceToVideoSceneMemory,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3SceneMemorySave": "Save Scene Memory",
    "H3SceneMemoryLoad": "Load Scene Memory",
    "H3SceneMemoryReferenceToVideo": "MiniMax H3 Reference to Video + Scene Memory",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
