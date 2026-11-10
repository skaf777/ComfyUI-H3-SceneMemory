"""Window selection and real adapter contract with no model inference."""
import sys
import types
import unittest

import numpy as np
import test_apply
from test_scene_memory import tensor


class SelectionTests(unittest.TestCase):
    setUp = test_apply.ApplyTests.setUp

    def select(self, sources, mode="Uniform", count=3, interval=17, manual="", pins="", render=None):
        module = sys.modules[self.Node.__module__.rsplit(".", 1)[0] + ".selection"]
        return module.select(sources, mode, count, 0, interval, manual, pins,
                             render or (lambda key: np.full((32, 32, 3), key[1] / 500, np.float32)))

    @staticmethod
    def source(slot, windows, identity=None):
        shape = (1, 24, (windows - 1) * 5 + 2, 4, 6)
        return (slot, types.SimpleNamespace(shape=shape), identity if identity is not None else f"source{slot}")

    def test_uniform_requested_counts_and_no_single_file_32_cap(self):
        for count in (1, 3, 6, 10, 15, 40):
            windows = 40 if count == 40 else 15
            with self.subTest(count=count):
                selected, warnings = self.select([self.source(1, windows)], count=count)
                self.assertEqual(len(selected), count)
                self.assertEqual(len(set(selected)), count)
                self.assertTrue(all(frame % 17 == 0 for _, frame in selected))
                self.assertFalse(warnings)
                if count > 1:
                    self.assertEqual((selected[0][1], selected[-1][1]), (0, 17 * (windows - 1)))

    def test_interval_alignment_duplicate_removal_and_manual(self):
        for interval in (5, 10, 17, 34, 68, 23):
            selected, _ = self.select([self.source(1, 15)], mode="Interval", count=10, interval=interval)
            self.assertEqual(len(selected), len(set(selected)))
            self.assertTrue(all(frame % 17 == 0 for _, frame in selected))
        selected, warnings = self.select([self.source(1, 15)], mode="Manual", count=5, manual="0, 5, 10, 187")
        self.assertEqual(selected, [(1, 0), (1, 187)])
        self.assertTrue(any("duplicate" in warning for warning in warnings))

    def test_pins_take_priority_and_use_multiple_explicit_files(self):
        sources = [self.source(1, 15), self.source(2, 15)]
        selected, warnings = self.select(sources, count=2, pins="1:0, 2:187, 2:190")
        self.assertEqual(selected, [(1, 0), (2, 187)])
        self.assertTrue(any("duplicate" in warning for warning in warnings))
        selected, warnings = self.select(sources, mode="Manual", count=1, pins="1:0, 2:187", manual="2:0")
        self.assertEqual(selected, [(1, 0), (2, 187)])
        self.assertTrue(any("priority" in warning for warning in warnings))
        selected, _ = self.select(sources, mode="Manual", count=2, manual="2:18, 1:5")
        self.assertEqual(selected, [(2, 17), (1, 0)])

    def test_multi_file_cap_and_identical_file_dedup(self):
        sources = [self.source(1, 40), self.source(2, 40)]
        selected, warnings = self.select(sources, count=70)
        self.assertEqual(len(selected), 32)
        self.assertTrue(any("at most 32" in w for w in warnings))
        duplicates = [self.source(1, 4, "same"), self.source(2, 4, "same")]
        selected, warnings = self.select(duplicates, mode="Manual", count=4, manual="1:0,2:0,1:17,2:17")
        self.assertEqual(selected, [(1, 0), (1, 17)])
        self.assertTrue(any("duplicate" in w for w in warnings))

    def test_smart_is_deterministic_deduplicates_and_keeps_pins(self):
        sources = [self.source(1, 6)]
        def render(key):
            n = key[1] // 17
            # Two almost identical moments and four distinct structures.
            n = 0 if n == 1 else n
            grid = np.zeros((32, 32, 3), dtype=np.float32)
            grid[:, (n * 5) % 24:(n * 5) % 24 + 8] = 0.8
            return grid
        first, warnings = self.select(sources, mode="Smart", count=6, pins="0", render=render)
        second, _ = self.select(sources, mode="Smart", count=6, pins="0", render=render)
        self.assertEqual(first, second)
        self.assertEqual(first[0], (1, 0))
        self.assertNotIn((1, 17), first)
        self.assertTrue(any("near-identical" in w for w in warnings))

    def test_bad_manual_or_pin_requests_fail_clearly(self):
        for pins in ("abc", "3:0", "99999"):
            with self.subTest(pins=pins), self.assertRaises(self.error):
                self.select([self.source(1, 15)], pins=pins)
        with self.assertRaises(self.error):
            self.select([self.source(1, 15)], mode="Manual")

    def test_distinct_sources_qwen_dit_order_and_preserved_audio(self):
        other = {"samples": tensor(np.full((1, 24, 7, 4, 6), 4))}
        before = self.video.numpy().copy()
        output = self.Node.execute(**self.args, enabled=True, scene_memory=self.memory, scene_memory_2=other,
                                   memory_mode="Manual", max_references=3, manual_frames="2:17, 1:0, 2:0")
        refs = output.args[0][0][1]["minimax_refs"][-3:]
        np.testing.assert_array_equal(refs[0]["latent"].numpy(), other["samples"].numpy()[:, :, 5:7])
        np.testing.assert_array_equal(refs[1]["latent"].numpy(), before[:, :, :2])
        np.testing.assert_array_equal(refs[2]["latent"].numpy(), other["samples"].numpy()[:, :, :2])
        np.testing.assert_array_equal(self.video.numpy(), before)
        self.assertEqual(output.ui["text"][0], "Selected scene memory: 2:17, 1:0, 2:0")
        self.assertEqual([item["type"] for item in self.clip.tokens["items"]][-3:], ["video"] * 3)
        self.assertEqual(output.args[0][0][1]["minimax_refs"][2]["audio"], "paired audio")
        self.assertEqual(output.args[0][0][1]["minimax_refs"][4]["value"], "new speech")

    def test_vae_cannot_mutate_saved_video_or_dit_payload(self):
        class MutatingVAE:
            def decode(self, window):
                window.numpy()[:] = 0
                return tensor(np.zeros((1, 5, 64, 96, 3)))
        original = self.video.numpy().copy()
        args = {**self.args, "vae": MutatingVAE()}
        output = self.Node.execute(**args, enabled=True, scene_memory=self.memory)
        np.testing.assert_array_equal(self.video.numpy(), original)
        np.testing.assert_array_equal(output.args[0][0][1]["minimax_refs"][-1]["latent"].numpy(), original[:, :, :2])

    def test_high_load_warning_is_reported(self):
        video = tensor(np.zeros((1, 24, 37, 4, 6)))
        output = self.Node.execute(**self.args, enabled=True, scene_memory={"samples": video},
                                   memory_mode="Uniform", max_references=8)
        self.assertTrue(any("heuristic" in message for message in output.ui["text"]))

    def test_multi_window_conditioning_and_smart_cache(self):
        class DiverseVAE:
            def __init__(self): self.count = 0
            def decode(self, z):
                self.count += 1
                number = int(z.numpy()[0, 0, 0, 0, 0])
                frame = np.zeros((1, 5, 64, 96, 3), dtype=np.float32)
                frame[:, :, :, (number * 3) % 64:(number * 3) % 64 + 24, :] = 0.8
                return tensor(frame)
        vae = DiverseVAE()
        source = np.zeros((1, 24, 17, 4, 6), np.float32)
        for t in range(17): source[:, :, t] = t
        memory = {"samples": tensor(source)}
        args = {**self.args, "vae": vae}
        out = self.Node.execute(**args, enabled=True, scene_memory=memory, scene_memory_2=memory,
                                memory_mode="Smart", max_references=3, pinned_frames="1:0")
        # The two identical sources represent the same windows; four candidates
        # are decoded once each, and selected windows are reused for Qwen/DiT.
        self.assertEqual(vae.count, 4)
        refs = out.args[0][0][1]["minimax_refs"]
        self.assertEqual(len(refs), 9)
        self.assertEqual([r["kind"] for r in refs[-3:]], ["video"] * 3)
        self.assertEqual(len([x for x in self.clip.tokens["items"] if x["type"] == "video"]), 5)
        self.assertIn("<Video 3>", self.clip.tokens["prompt"])
        self.assertIn("<Video 5>", self.clip.tokens["prompt"])
        self.assertEqual(refs[2]["audio"], "paired audio")
        self.assertEqual(refs[4]["value"], "new speech")
        self.assertIn("1:0", out.ui["text"][0])
        for ref in refs[-3:]: self.assertEqual(ref["ref_audio_t"], 0)


if __name__ == "__main__": unittest.main(verbosity=2)
