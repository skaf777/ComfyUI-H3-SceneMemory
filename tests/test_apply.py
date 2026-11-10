"""Contract tests with Comfy/H3 doubles; no model inference is performed."""
import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import comfy_stubs
from test_scene_memory import tensor

ROOT = pathlib.Path(__file__).resolve().parents[1]


class Clip:
    def __init__(self, tags=True):
        self.tokens = None
        self.tags = tags
    def tokenize(self, prompt, **kwargs):
        self.tokens = {"prompt": prompt, "items": kwargs["minimax_ref_items"]}
        return self.tokens
    def encode_from_tokens_scheduled(self, tokens):
        meta = {"existing_meta": "preserved", "minimax_keyframes": ["original guide"]}
        if self.tags: meta["minimax_token_tags"] = "tags"
        return [["embedding", meta]]


class VideoVAE:
    def __init__(self, bad=False): self.windows = []; self.bad = bad
    def decode(self, window):
        self.windows.append(window)
        n = 4 if self.bad else 5
        return tensor(np.zeros((1, n, window.shape[3] * 16, window.shape[4] * 16, 3)))


class ApplyTests(unittest.TestCase):
    def setUp(self):
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_output_directory = lambda: "/tmp"
        self.patcher = patch.dict(sys.modules, {**comfy_stubs.modules(), "folder_paths": folder_paths})
        self.patcher.start(); self.addCleanup(self.patcher.stop)
        name = "h3_scene_apply_tests"
        spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
        package = importlib.util.module_from_spec(spec); sys.modules[name] = package
        self.addCleanup(lambda: [sys.modules.pop(key, None) for key in list(sys.modules) if key == name or key.startswith(name + ".")])
        spec.loader.exec_module(package)
        self.Node = package.NODE_CLASS_MAPPINGS["H3SceneMemoryReferenceToVideo"]
        self.error = sys.modules[name + ".scene_memory"].SceneMemoryError
        self.clip = Clip(); self.vae = VideoVAE()
        self.video = tensor(np.arange(1 * 24 * 7 * 4 * 6).reshape(1, 24, 7, 4, 6))
        self.memory = {"samples": self.video}
        self.args = dict(clip=self.clip, vae=self.vae, audio_vae="audio VAE", prompt="new camera and dialogue",
                         width=96, height=64, length=22, ref_image_size="max",
                         ref_images={"ref_image_0": "image A", "ref_image_1": "image B"},
                         ref_videos={"ref_video_0": "video A", "ref_video_1": "video B"},
                         ref_video_audios={"ref_video_audio_0": "paired audio"},
                         ref_audios={"ref_audio_0": "new speech", "ref_audio_1": "sound"})

    def test_schema_extends_original_without_changing_inputs_or_outputs(self):
        original = comfy_stubs.Core.define_schema(); new = self.Node.define_schema()
        self.assertEqual([i.name for i in new.inputs[:len(original.inputs)]], [i.name for i in original.inputs])
        self.assertEqual(new.outputs, original.outputs)
        self.assertEqual([i.name for i in new.inputs[len(original.inputs):len(original.inputs) + 3]], ["enabled", "memory_frame", "scene_memory"])
        self.assertTrue(new.inputs[len(original.inputs) + 2].kwargs["lazy"])
        self.assertEqual(self.Node.check_lazy_status(False), [])
        self.assertEqual(self.Node.check_lazy_status(True, scene_memory=None), ["scene_memory"])
        self.assertEqual(self.Node.check_lazy_status(True, scene_memory=self.memory), [])

    def test_disabled_delegates_to_core_and_ignores_bad_memory(self):
        out = self.Node.execute(**self.args, enabled=False, scene_memory={"invalid": True}, memory_frame=-1)
        self.assertIs(out, comfy_stubs.Core.last_result)
        self.assertEqual(comfy_stubs.Core.last_args, self.args)
        self.assertEqual(self.vae.windows, [])
        self.assertEqual(self.clip.tokens["prompt"], self.args["prompt"])
        self.assertEqual([r["kind"] for r in out.args[0][0][1]["minimax_refs"]], ["image", "image", "video_audio", "video", "audio", "audio"])

    def test_enabled_appends_qwen_and_latent_in_matching_order(self):
        out = self.Node.execute(**self.args, enabled=True, scene_memory=self.memory, memory_frame=21)
        positive, target = out.args
        self.assertEqual(target, {"samples": "target AV latent"})
        self.assertEqual([item["type"] for item in self.clip.tokens["items"]],
                         ["image", "image", "audio", "video", "video", "audio", "audio", "video"])
        self.assertIn("<Video 3>", self.clip.tokens["prompt"])
        self.assertTrue(self.clip.tokens["prompt"].startswith(self.args["prompt"]))
        presented = self.clip.tokens["items"][-1]
        self.assertEqual(presented["data"].shape, (1, 64, 96, 3))
        self.assertEqual(presented["timestamps"], [0.0])
        metadata = positive[0][1]
        self.assertEqual(metadata["minimax_token_tags"], "tags")
        self.assertEqual(metadata["minimax_keyframes"], ["original guide"])
        self.assertEqual(metadata["existing_meta"], "preserved")
        refs = metadata["minimax_refs"]
        self.assertEqual([r["kind"] for r in refs], ["image", "image", "video_audio", "video", "audio", "audio", "video"])
        self.assertEqual(refs[2]["audio"], "paired audio")
        self.assertEqual(refs[4]["value"], "new speech")
        block = refs[-1]
        self.assertEqual((block["latent_t"], block["latent_h"], block["latent_w"], block["ref_audio_t"]), (2, 4, 6, 0))
        self.assertIsNone(block["audio_latent"])
        np.testing.assert_array_equal(block["latent"].numpy(), self.video.numpy()[:, :, 5:7])
        np.testing.assert_array_equal(self.vae.windows[0].numpy(), block["latent"].numpy())

    def test_requires_valid_memory_vae_frame_and_h3_encoder(self):
        for extra in ({"scene_memory": None}, {"scene_memory": {}}, {"scene_memory": self.memory, "vae": None},
                      {"scene_memory": self.memory, "memory_frame": 22}, {"scene_memory": self.memory, "memory_frame": -1},
                      {"scene_memory": self.memory, "vae": VideoVAE(bad=True)},
                      {"scene_memory": self.memory, "clip": Clip(tags=False)}):
            kwargs = {**self.args, **extra, "enabled": True}
            with self.subTest(extra=extra), self.assertRaises(self.error):
                self.Node.execute(**kwargs)

    def test_no_external_refs_numbers_memory_as_video_one(self):
        args = {**self.args, "ref_images": None, "ref_videos": None, "ref_audios": None, "ref_video_audios": None}
        result = self.Node.execute(**args, enabled=True, scene_memory=self.memory)
        self.assertIn("<Video 1>", self.clip.tokens["prompt"])
        self.assertEqual(len(result.args[0][0][1]["minimax_refs"]), 1)


class StockSourceContractTests(unittest.TestCase):
    """Exercise the installed official execute method with lightweight dependencies.

    Set COMFYUI_H3_SOURCE to nodes_minimax_h3.py. This tests the actual source
    control flow, but does not run Qwen, VAE, PyTorch or the diffusion model.
    """
    setUp = ApplyTests.setUp

    def test_official_execute_source_with_all_reference_types(self):
        import ast
        import math
        import os
        source = os.environ.get("COMFYUI_H3_SOURCE")
        if not source:
            self.skipTest("set COMFYUI_H3_SOURCE to the installed ComfyUI nodes_minimax_h3.py")
        tree = ast.parse(pathlib.Path(source).read_text())
        core_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3ReferenceToVideo")
        method = next(n for n in core_class.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
        module = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name="StockCore", bases=[], keywords=[], body=[method], decorator_list=[])], type_ignores=[]))

        class FakeNodeHelpers:
            @staticmethod
            def conditioning_set_values(conditioning, values):
                return [[embedding, {**meta, **values}] for embedding, meta in conditioning]

        class WorkingVAE(VideoVAE):
            def encode(self, image):
                return tensor(np.zeros((1, 24, 2, image.shape[1] // 16, image.shape[2] // 16)))

        class AudioLatent:
            shape = (1, 32, 2, 4)

        fake_io = types.SimpleNamespace(NodeOutput=comfy_stubs.NodeOutput)
        ns = dict(io=fake_io, math=math, node_helpers=FakeNodeHelpers, CANVAS_MULTIPLE=32,
                  REF_IMAGE_SHORT_EDGE=2048, FPS=24,
                  _empty_av_latent=lambda width, height, length: ({"samples": "target AV latent"}, length),
                  _resize=lambda frames, width, height, crop: frames,
                  adapt_canvas=lambda width, height: (width, height),
                  _encode_ref_audio=lambda vae, audio: (AudioLatent(), 4))
        exec(compile(module, source, "exec"), ns)
        original_core = sys.modules[self.Node.__module__].h3.MiniMaxH3ReferenceToVideo
        sys.modules[self.Node.__module__].h3.MiniMaxH3ReferenceToVideo = ns["StockCore"]
        self.addCleanup(setattr, sys.modules[self.Node.__module__].h3, "MiniMaxH3ReferenceToVideo", original_core)
        image = tensor(np.zeros((1, 64, 96, 3)))
        video = tensor(np.zeros((5, 64, 96, 3)))
        args = {**self.args, "vae": WorkingVAE(), "ref_images": {"ref_image_1": image},
                "ref_videos": {"ref_video_1": video}, "ref_video_audios": {"ref_video_audio_1": "audio"},
                "ref_audios": {"ref_audio_1": "speech"}}
        result = self.Node.execute(**args, enabled=True, scene_memory=self.memory)
        refs = result.args[0][0][1]["minimax_refs"]
        self.assertEqual([r["kind"] for r in refs], ["image", "video_audio", "audio", "video"])
        self.assertEqual([i["type"] for i in self.clip.tokens["items"]], ["image", "audio", "video", "audio", "video"])
        self.assertEqual(refs[-1]["ref_audio_t"], 0)
        self.assertIn("<Video 2>", self.clip.tokens["prompt"])
        result = self.Node.execute(**args, enabled=True, scene_memory=self.memory,
                                   memory_mode="Uniform", max_references=2)
        refs = result.args[0][0][1]["minimax_refs"]
        self.assertEqual([r["kind"] for r in refs], ["image", "video_audio", "audio", "video", "video"])
        self.assertEqual([i["type"] for i in self.clip.tokens["items"]],
                         ["image", "audio", "video", "audio", "video", "video"])
        self.assertIn("<Video 3>", self.clip.tokens["prompt"])
        np.testing.assert_array_equal(refs[-2]["latent"].numpy(), self.video.numpy()[:, :, :2])
        np.testing.assert_array_equal(refs[-1]["latent"].numpy(), self.video.numpy()[:, :, 5:7])


if __name__ == "__main__": unittest.main(verbosity=2)
