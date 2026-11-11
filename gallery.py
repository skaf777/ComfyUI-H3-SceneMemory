"""Read-only gallery endpoints for scene memory files and their thumbnails."""
import os
import re

from aiohttp import web

from .nodes import memory_folder
from .scene_memory import SceneMemoryError, _path, available_memories


def entries(folder):
    result = []
    for filename in available_memories(folder):
        stem = filename[:-len('.safetensors')]
        match = re.fullmatch(r'(.*)_(\d{4,})', stem)
        png = os.path.join(os.path.realpath(folder), stem + '.png')
        preview = (os.path.dirname(os.path.realpath(png)) == os.path.realpath(folder)
                   and os.path.isfile(png))
        result.append({'filename': filename, 'scene': match.group(1) if match else stem,
                       'take': match.group(2) if match else None, 'preview': preview})
    return result


def register_routes(routes):
    @routes.get('/h3_scene_memory/list')
    async def list_memories(request):
        return web.json_response(entries(memory_folder()))

    @routes.get('/h3_scene_memory/thumbnail')
    async def thumbnail(request):
        folder = memory_folder()
        try:
            filename = request.query.get('filename', '')
            path = _path(folder, filename)
            if not filename.lower().endswith('.safetensors') or not os.path.isfile(path):
                raise web.HTTPNotFound()
            png = os.path.splitext(path)[0] + '.png'
            if os.path.dirname(os.path.realpath(png)) != os.path.realpath(folder) or not os.path.isfile(png):
                raise web.HTTPNotFound()
            return web.FileResponse(png, headers={'Cache-Control': 'no-cache'})
        except SceneMemoryError:
            raise web.HTTPNotFound()
