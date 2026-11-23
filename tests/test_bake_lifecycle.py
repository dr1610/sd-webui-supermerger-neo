import importlib.util
import inspect
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch

source = Path(__file__).parents[1] / "scripts/mergers/bake_lifecycle.py"
spec = importlib.util.spec_from_file_location("bake_lifecycle_tested", source)
lifecycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lifecycle)


class BakeLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.lock = threading.Lock()
        self.model_data = types.SimpleNamespace(sd_model="original", forge_hash="original-hash", forge_loading_parameters={})
        self.processing = type("Processing", (), {"cached_c": ["old positive"], "cached_uc": ["old negative"]})
        self.networks = types.SimpleNamespace(loaded_networks=["old module"], networks_in_memory={"old": "model ref"})
        self.lora = types.SimpleNamespace(loaded_loras=["old lora"])
        self.sd_models = types.SimpleNamespace(forge_model_reload=lambda: None, model_data=self.model_data,
                                              checkpoints_loaded={"old": "model"}, unload_model_weights=self.unload)
        modules = {
            "modules": types.SimpleNamespace(sd_models=self.sd_models),
            "modules.call_queue": types.SimpleNamespace(queue_lock=self.lock),
            "modules.processing": types.SimpleNamespace(StableDiffusionProcessing=self.processing),
            "modules_forge.main_entry": types.SimpleNamespace(refresh_model_loading_parameters=self.refresh),
            "scripts.A1111": types.SimpleNamespace(networks=self.networks),
            "lora": self.lora,
        }
        self.stub = patch.dict(sys.modules, modules)
        self.stub.start()
        self.addCleanup(self.stub.stop)

    def unload(self):
        self.assertTrue(self.lock.locked())
        self.events.append("unload")
        self.model_data.sd_model = "unloaded"

    def refresh(self):
        self.assertTrue(self.lock.locked())
        self.events.append("refresh")
        self.model_data.forge_loading_parameters = {"checkpoint": "selected", "vae": "selected-vae"}

    def assert_clean(self):
        self.assertEqual(self.events, ["unload", "refresh"])
        self.assertEqual(self.model_data.sd_model, "unloaded")
        self.assertEqual(self.model_data.forge_hash, "")
        self.assertEqual(self.model_data.forge_loading_parameters["checkpoint"], "selected")
        self.assertEqual(self.processing.cached_c, [None]*3)
        self.assertEqual(self.processing.cached_uc, [None]*3)
        self.assertEqual(self.networks.loaded_networks, [])
        self.assertEqual(self.networks.networks_in_memory, {})
        self.assertEqual(self.lora.loaded_loras, [])
        self.assertEqual(self.sd_models.checkpoints_loaded, {})
        self.assertFalse(self.lock.locked())

    def test_success_keeps_result_and_signature(self):
        @lifecycle.protect_bake
        def bake(model, ratio=1):
            self.assertTrue(self.lock.locked())
            self.model_data.sd_model = "temporary"
            return (model, ratio, "saved")
        self.assertEqual(bake("example", ratio=0.7), ("example", 0.7, "saved"))
        self.assertEqual(bake.__name__, "bake")
        self.assertEqual(str(inspect.signature(bake)), "(model, ratio=1)")
        self.assert_clean()

    def test_merge_and_save_exceptions_both_cleanup(self):
        for phase in ["merge", "save"]:
            self.events.clear()
            @lifecycle.protect_bake
            def bake():
                self.model_data.sd_model = "partly modified"
                raise RuntimeError(phase)
            with self.assertRaisesRegex(RuntimeError, phase):
                bake()
            self.assert_clean()

    def test_error_cleanup_does_not_hide_bake_error(self):
        def fail_unload():
            raise ValueError("cleanup failure")
        self.sd_models.unload_model_weights = fail_unload
        @lifecycle.protect_bake
        def bake():
            raise RuntimeError("original failure")
        with self.assertLogs(level="ERROR"), self.assertRaisesRegex(RuntimeError, "original failure"):
            bake()
        self.assertEqual(self.model_data.forge_hash, "")
        self.assertEqual(self.events, ["refresh"])
        self.assertFalse(self.lock.locked())

    def test_successful_bake_surfaces_cleanup_failure(self):
        self.sd_models.unload_model_weights = lambda: (_ for _ in ()).throw(ValueError("cleanup"))
        with self.assertRaisesRegex(ValueError, "cleanup"):
            lifecycle.protect_bake(lambda: "saved")()
        self.assertFalse(self.lock.locked())

    def test_bake_waits_for_generation_lock(self):
        started, entered, finished = threading.Event(), threading.Event(), threading.Event()
        @lifecycle.protect_bake
        def bake():
            entered.set()
        def run():
            started.set()
            bake()
            finished.set()
        self.lock.acquire()
        thread = threading.Thread(target=run)
        thread.start()
        try:
            self.assertTrue(started.wait(1))
            self.assertFalse(entered.wait(0.05))
        finally:
            self.lock.release()
        self.assertTrue(finished.wait(2))
        thread.join()
        self.assert_clean()

    def test_legacy_host_untouched(self):
        self.sd_models.read_state_dict = lambda: None
        self.assertEqual(lifecycle.protect_bake(lambda: "legacy")(), "legacy")
        self.assertEqual(self.events, [])
        self.assertEqual(self.networks.loaded_networks, ["old module"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
