"""Storage contract tests. Use real PyTorch when installed, otherwise a numpy test double.

Both modes use the real safetensors serializer. The test double allows lightweight
CI to exercise the file format but does not replace a test in ComfyUI/PyTorch.
"""
import importlib
import os
import pathlib
import sys
import tempfile
import types
import unittest

import numpy as np
from PIL import PngImagePlugin  # Keep Pillow's PNG plugin registered across module stubs.

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    REAL_TORCH = True
except ImportError:
    REAL_TORCH = False
    class Tensor:
        is_nested = False
        def __init__(self, array):
            self.array = np.asarray(array)
        @property
        def shape(self): return self.array.shape
        @property
        def ndim(self): return self.array.ndim
        @property
        def dtype(self): return self.array.dtype
        def detach(self): return self
        def to(self, device=None, copy=False, dtype=None): return Tensor(self.array.astype(dtype, copy=copy) if dtype else (self.array.copy() if copy else self.array))
        def contiguous(self): return Tensor(np.ascontiguousarray(self.array))
        def view(self, dtype): return Tensor(self.array.view(dtype))
        def reshape(self, *shape): return Tensor(self.array.reshape(*shape))
        def numpy(self): return self.array
        def __getitem__(self, key): return Tensor(self.array[key])
        def clone(self): return Tensor(self.array.copy())
    torch = types.ModuleType("torch")
    torch.Tensor = Tensor
    torch.float16 = np.dtype("float16")
    torch.bfloat16 = object()
    torch.float32 = np.dtype("float32")
    torch.uint8 = np.uint8
    torch.from_numpy = Tensor
    torch.isfinite = lambda value: np.isfinite(value.array)
    sys.modules["torch"] = torch
    import safetensors.numpy as sn
    st = types.ModuleType("safetensors.torch")
    st.save_file = lambda tensors, filename, metadata=None: sn.save_file(
        {k: v.array for k, v in tensors.items()}, filename, metadata=metadata)
    st.load_file = lambda filename, device=None: {k: Tensor(v) for k, v in sn.load_file(filename).items()}
    sys.modules["safetensors.torch"] = st

import scene_memory as sm
if not REAL_TORCH:
    from safetensors import safe_open as real_safe_open
    sm.safe_open = lambda path, framework, device: real_safe_open(path, framework="np", device="cpu")


def tensor(array):
    return torch.from_numpy(np.array(array, dtype=np.float32))


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = self.tmp.name
        self.values = np.arange(1 * 24 * 7 * 4 * 6, dtype=np.float32).reshape(1, 24, 7, 4, 6) / 100
        self.latent = {"samples": tensor(self.values)}

    def test_roundtrip_and_metadata(self):
        path = sm.save_memory(self.folder, "scene", self.latent)
        self.assertEqual(os.path.basename(path), "scene.safetensors")
        result = sm.load_memory(self.folder, "scene.safetensors")
        np.testing.assert_array_equal(result["samples"].numpy(), self.values)
        self.assertEqual(result["samples"].dtype, self.latent["samples"].dtype)
        with sm.safe_open(path, framework="pt", device="cpu") as f:
            meta = f.metadata()
            self.assertEqual(list(f.keys()), ["video_latent"])
            self.assertEqual((meta["width"], meta["height"], meta["latent_frames"], meta["frame_count"]), ("96", "64", "7", "22"))
            self.assertEqual((meta["source"], meta["dtype"], meta["fps"]), ("stage1", "float32", "24"))
            self.assertEqual(meta["shape"], "[1,24,7,4,6]")

    def test_no_overwrite(self):
        path = sm.save_memory(self.folder, "scene", self.latent)
        before = pathlib.Path(path).read_bytes()
        with self.assertRaisesRegex(sm.SceneMemoryError, "already exists"):
            sm.save_memory(self.folder, "scene", self.latent)
        self.assertEqual(pathlib.Path(path).read_bytes(), before)
        self.assertEqual(os.listdir(self.folder), ["scene.safetensors"])

    def test_corrupt_tensor_bytes_detected(self):
        path = sm.save_memory(self.folder, "scene", self.latent)
        data = bytearray(pathlib.Path(path).read_bytes())
        data[-1] ^= 1
        pathlib.Path(path).write_bytes(data)
        with self.assertRaisesRegex(sm.SceneMemoryError, "checksum mismatch"):
            sm.load_memory(self.folder, "scene")

    def test_truncated_and_invalid_file(self):
        path = sm.save_memory(self.folder, "scene", self.latent)
        data = pathlib.Path(path).read_bytes()
        pathlib.Path(path).write_bytes(data[:40])
        with self.assertRaises(sm.SceneMemoryError):
            sm.load_memory(self.folder, "scene")
        pathlib.Path(path).write_bytes(b"not safetensors")
        with self.assertRaises(sm.SceneMemoryError):
            sm.load_memory(self.folder, "scene")

    def test_wrong_metadata_and_tensor_names(self):
        path = sm.save_memory(self.folder, "scene", self.latent)
        with sm.safe_open(path, framework="pt", device="cpu") as f:
            meta = dict(f.metadata())
        meta["frame_count"] = "999"
        sm.save_file({sm.TENSOR_NAME: self.latent["samples"]}, path, metadata=meta)
        with self.assertRaisesRegex(sm.SceneMemoryError, "frame_count"):
            sm.load_memory(self.folder, "scene")
        sm.save_file({"wrong": self.latent["samples"]}, path, metadata=meta)
        with self.assertRaisesRegex(sm.SceneMemoryError, "exactly one"):
            sm.load_memory(self.folder, "scene")

    def test_invalid_latents(self):
        cases = [np.zeros((1, 32, 7, 4, 6)), np.zeros((1, 24, 3, 4, 6)),
                 np.zeros((2, 24, 7, 4, 6)), np.zeros((1, 24, 7, 3, 6)),
                 np.full((1, 24, 2, 4, 6), np.nan)]
        for arr in cases:
            with self.subTest(shape=arr.shape), self.assertRaises(sm.SceneMemoryError):
                sm.save_memory(self.folder, "bad", {"samples": tensor(arr)})
        with self.assertRaises(sm.SceneMemoryError):
            sm.save_memory(self.folder, "bad", {})

    def test_filename_and_fingerprint(self):
        for name in ("../escape", "a/b", "a\\b", ""):
            with self.subTest(name=name), self.assertRaises(sm.SceneMemoryError):
                sm.save_memory(self.folder, name, self.latent)
        sm.save_memory(self.folder, "scene", self.latent)
        self.assertEqual(sm.available_memories(self.folder), ["scene.safetensors"])
        old = sm.file_fingerprint(self.folder, "scene.safetensors")
        path = pathlib.Path(self.folder, "scene.safetensors")
        data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
        self.assertNotEqual(old, sm.file_fingerprint(self.folder, "scene.safetensors"))

    @unittest.skipUnless(REAL_TORCH, "requires real PyTorch bfloat16")
    def test_bfloat16_bit_exact(self):
        video = torch.from_numpy(self.values).to(dtype=torch.bfloat16)
        sm.save_memory(self.folder, "bf16", {"samples": video})
        result = sm.load_memory(self.folder, "bf16")["samples"]
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(video.view(torch.uint8), result.view(torch.uint8)))


class NodeRegistrationTests(unittest.TestCase):
    def test_nodes_register_and_roundtrip(self):
        import importlib.util
        with tempfile.TemporaryDirectory() as folder:
            previous = sys.modules.get("folder_paths")
            folder_paths = types.ModuleType("folder_paths")
            folder_paths.get_output_directory = lambda: folder
            sys.modules["folder_paths"] = folder_paths
            self.addCleanup(lambda: sys.modules.pop("folder_paths", None) if previous is None else sys.modules.__setitem__("folder_paths", previous))
            spec = importlib.util.spec_from_file_location(
                "h3_scene_memory_test_package", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
            package = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = package
            self.addCleanup(lambda: [sys.modules.pop(key, None) for key in list(sys.modules) if key == spec.name or key.startswith(spec.name + ".")])
            from unittest.mock import patch
            import comfy_stubs
            with patch.dict(sys.modules, comfy_stubs.modules()):
                spec.loader.exec_module(package)
                if not REAL_TORCH:
                    sys.modules[spec.name + ".scene_memory"].safe_open = sm.safe_open
            self.assertEqual(set(package.NODE_CLASS_MAPPINGS), {"H3SceneMemorySave", "H3SceneMemoryLoad", "H3SceneMemoryReferenceToVideo", "H3SceneMemoryTimeline"})
            Save = package.NODE_CLASS_MAPPINGS["H3SceneMemorySave"]
            Load = package.NODE_CLASS_MAPPINGS["H3SceneMemoryLoad"]
            original = {"samples": tensor(np.zeros((1, 24, 2, 4, 6)))}
            passthrough, saved_path = Save().save(original, "living_room")
            self.assertIs(passthrough, original)
            self.assertTrue(os.path.isfile(saved_path))
            self.assertEqual(Load.INPUT_TYPES()["required"]["filename"][0], ["living_room.safetensors"])
            loaded, = Load().load("living_room.safetensors")
            np.testing.assert_array_equal(loaded["samples"].numpy(), original["samples"].numpy())
            from test_apply import VideoVAE
            vae = VideoVAE()
            _, numbered = Save().save(original, "room", auto_numbering=True, vae=vae)
            self.assertEqual(os.path.basename(numbered), "room_0001.safetensors")
            self.assertEqual(len(vae.windows), 1)
            self.assertEqual(vae.windows[0].shape[2], 2)
            from PIL import Image
            with Image.open(os.path.splitext(numbered)[0] + ".png") as preview:
                self.assertEqual(preview.format, "PNG")
                self.assertLessEqual(preview.width, 256)
            _, failed_preview = Save().save(original, "failed", vae=VideoVAE(bad=True))
            self.assertTrue(os.path.isfile(failed_preview))
            self.assertFalse(os.path.exists(os.path.splitext(failed_preview)[0] + ".png"))
            with patch.dict(sys.modules, comfy_stubs.modules()):
                gallery = importlib.import_module(spec.name + ".gallery")
                items = {item['filename']: item for item in gallery.entries(os.path.join(folder, 'scene_memory'))}
            self.assertFalse(items['living_room.safetensors']['preview'])
            self.assertEqual((items['room_0001.safetensors']['scene'], items['room_0001.safetensors']['take'], items['room_0001.safetensors']['preview']), ('room', '0001', True))


if __name__ == "__main__":
    unittest.main(verbosity=2)

class NumberingTests(unittest.TestCase):
    setUp = StorageTests.setUp
    def test_numbering_continues_from_disk_and_does_not_overwrite(self):
        first = sm.save_memory(self.folder, 'Сцена [1].safetensors', self.latent, True)
        before = pathlib.Path(first).read_bytes()
        pathlib.Path(self.folder, 'Сцена [1]_0010.safetensors').write_bytes(b'old')
        safe_open = sm.safe_open
        try:
            second = importlib.reload(sm).save_memory(self.folder, 'Сцена [1]', self.latent, True)
        finally:
            sm.safe_open = safe_open
        self.assertEqual(os.path.basename(first), 'Сцена [1]_0001.safetensors')
        self.assertEqual(os.path.basename(second), 'Сцена [1]_0011.safetensors')
        self.assertEqual(pathlib.Path(first).read_bytes(), before)
        self.assertEqual(pathlib.Path(self.folder, 'Сцена [1]_0010.safetensors').read_bytes(), b'old')

    def test_concurrent_saves_are_unique(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as pool:
            paths = list(pool.map(lambda _: sm.save_memory(self.folder, 'take', self.latent, True), range(20)))
        self.assertEqual({os.path.basename(path) for path in paths},
                         {f'take_{i:04}.safetensors' for i in range(1, 21)})
