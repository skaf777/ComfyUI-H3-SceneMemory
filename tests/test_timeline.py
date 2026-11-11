"""Timeline contract tests using lightweight Comfy and VAE doubles."""
import json
import unittest
import numpy as np
from test_apply import ApplyTests, VideoVAE
from test_scene_memory import tensor


class TimelineTests(ApplyTests):
    def setUp(self):
        super().setUp()
        import sys
        self.timeline = sys.modules[self.Node.__module__.rsplit('.', 1)[0] + '.timeline']
        self.Timeline = self.timeline.SceneMemoryTimeline()
        self.other = {"samples": tensor(np.ones((1, 24, 7, 4, 6)))}

    def test_preview_cache_and_persisted_selection(self):
        first = self.Timeline.execute(self.memory, self.vae, scene_memory_2=self.other, preview_request="one")
        images = first['ui']['previews']
        self.assertEqual([(i['slot'], i['frame']) for i in images], [(1, 0), (1, 17), (2, 0), (2, 17)])
        self.assertEqual(len(self.vae.windows), 4)
        selection = {"version": 1, "items": [{k: images[1][k] for k in ('slot', 'frame', 'source')}]}
        restored = self.Timeline.execute(self.memory, self.vae, selection=json.dumps(selection), scene_memory_2=self.other, preview_request="two")
        self.assertEqual(restored['result'][0], selection)
        self.assertEqual(len(self.vae.windows), 4)
        stale = self.Timeline.execute(self.other, self.vae, selection=json.dumps(selection), preview_request="three")
        self.assertTrue(stale['ui']['warning'])

    def test_normal_execution_only_returns_selection_without_decoding_or_loading(self):
        saved = {"version": 1, "items": [{"slot": 1, "frame": 17, "source": "saved"}]}
        result = self.Timeline.execute(selection=json.dumps(saved))
        self.assertEqual(result, (saved,))
        self.assertEqual(self.vae.windows, [])
        self.assertEqual(self.Timeline.check_lazy_status(scene_memory=None, vae=None), [])
        self.assertEqual(self.Timeline.check_lazy_status(preview_request="button", scene_memory=None,
                                                        vae=None, scene_memory_2=None),
                         ["scene_memory", "vae", "scene_memory_2"])
        # Supplying inputs during a regular graph execution must not decode either.
        self.assertEqual(self.Timeline.execute(self.memory, self.vae, selection=json.dumps(saved)), (saved,))
        self.assertEqual(self.vae.windows, [])

    def test_user_mode_passes_selected_windows_in_order_and_rejects_mismatch(self):
        sources = self.timeline.sources_from_memories((self.memory, self.other))
        items = [{"slot": 2, "frame": 17, "source": sources[1][2]}, {"slot": 1, "frame": 0, "source": sources[0][2]}]
        selection = {"version": 1, "items": items + items[:1]}
        out = self.Node.execute(**self.args, enabled=True, scene_memory=self.memory, scene_memory_2=self.other,
                                memory_mode="User", user_selection=selection)
        refs = out.args[0][0][1]['minimax_refs'][-2:]
        np.testing.assert_array_equal(refs[0]['latent'].numpy(), self.other['samples'].numpy()[:, :, 5:7])
        np.testing.assert_array_equal(refs[1]['latent'].numpy(), self.memory['samples'].numpy()[:, :, :2])
        self.assertEqual(len(self.clip.tokens['items'][-2:]), 2)
        with self.assertRaisesRegex(self.error, 'source mismatch'):
            self.Node.execute(**self.args, enabled=True, scene_memory=self.other, scene_memory_2=self.memory,
                              memory_mode="User", user_selection=selection)
        with self.assertRaisesRegex(self.error, 'connected Scene Memory Timeline'):
            self.Node.execute(**self.args, enabled=True, scene_memory=self.memory, memory_mode="User")
        self.assertEqual(self.Node.check_lazy_status(True, memory_mode='User', user_selection=None), ['user_selection'])
        self.assertEqual(self.Node.check_lazy_status(True, memory_mode='Smart', user_selection=None), [])
