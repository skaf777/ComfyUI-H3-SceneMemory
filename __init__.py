"""ComfyUI-H3-SceneMemory custom nodes."""

from .nodes import LoadSceneMemory, SaveSceneMemory
from .reference_node import MiniMaxH3ReferenceToVideoSceneMemory
from .timeline import SceneMemoryTimeline

NODE_CLASS_MAPPINGS = {
    "H3SceneMemoryTimeline": SceneMemoryTimeline,
    "H3SceneMemorySave": SaveSceneMemory,
    "H3SceneMemoryLoad": LoadSceneMemory,
    "H3SceneMemoryReferenceToVideo": MiniMaxH3ReferenceToVideoSceneMemory,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3SceneMemoryTimeline": "Scene Memory Timeline",
    "H3SceneMemorySave": "Save Scene Memory",
    "H3SceneMemoryLoad": "Load Scene Memory",
    "H3SceneMemoryReferenceToVideo": "MiniMax H3 Reference to Video + Scene Memory",
}

WEB_DIRECTORY = "./web"

# Routes are registered when loaded by ComfyUI; unit tests can import the
# package without starting a server.
try:
    from server import PromptServer
except ImportError:
    PromptServer = None
if PromptServer is not None and PromptServer.instance is not None:
    from . import gallery
    gallery.register_routes(PromptServer.instance.routes)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
