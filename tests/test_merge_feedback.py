import ast
import os
import pathlib
import threading
import traceback
import unittest
from collections import deque
from queue import Empty, Queue


ROOT = pathlib.Path(__file__).resolve().parents[1] / "scripts"


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class FakeModels:
    def __init__(self):
        self.available = {}
        self.refresh_count = 0

    def get_closet_checkpoint_match(self, name):
        return self.available.get(name)

    def list_models(self):
        self.refresh_count += 1
        self.available["new"] = type("Info", (), {"filename": __file__})()


class MergeFeedbackTests(unittest.TestCase):
    def test_merge_only_defaults_are_accepted_by_ratio_parser(self):
        source = ast.parse((ROOT / "GenParamGetter.py").read_text(encoding="utf-8-sig"))
        defaults = next(
            ast.literal_eval(node.value)
            for node in source.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_MERGE_ONLY_GENERATION_DEFAULTS" for target in node.targets)
        )
        mode, settings, upper, lower, upper_d, lower_d, seed, count, custom, digits = defaults[-10:]
        lucks = {"upp": upper, "low": lower, "uppd": upper_d, "lowd": lower_d, "round": digits}
        ratio_parser = load_function(
            ROOT / "mergers" / "mergers.py", "randdealer", {"RANDMAP": [0, 50, 100]}
        )
        self.assertEqual(ratio_parser("0.2", [0.5] * 300, 0, lucks, None), ("0.2", None))

    def test_missing_models_are_rejected_before_loading(self):
        models = FakeModels()
        namespace = {"sd_models": models, "os": os, "MODES": ["Weight", "Add", "Triple", "Twice"]}
        validate = load_function(ROOT / "mergers" / "mergers.py", "validate_merge_models", namespace)
        self.assertIn("Model A, B, C", validate(None, None, None, "Triple sum"))
        self.assertEqual(models.refresh_count, 0)

    def test_new_checkpoint_is_found_after_refresh(self):
        models = FakeModels()
        models.available["base"] = type("Info", (), {"filename": __file__})()
        namespace = {"sd_models": models, "os": os, "MODES": ["Weight", "Add", "Triple", "Twice"]}
        validate = load_function(ROOT / "mergers" / "mergers.py", "validate_merge_models", namespace)
        self.assertIsNone(validate("base", "base", "new", "Triple sum"))
        self.assertEqual(models.refresh_count, 1)

    def test_merge_status_streams_before_final_result(self):
        source = ast.parse((ROOT / "GenParamGetter.py").read_text(encoding="utf-8-sig"))
        defaults = next(
            ast.literal_eval(node.value)
            for node in source.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_MERGE_ONLY_GENERATION_DEFAULTS" for target in node.targets)
        )
        namespace = {
            "deque": deque,
            "Empty": Empty,
            "Queue": Queue,
            "Event": threading.Event,
            "Lock": threading.Lock,
            "Thread": threading.Thread,
            "traceback": traceback,
            "cachedealer": lambda start: None,
            "_MERGE_ONLY_GENERATION_DEFAULTS": defaults,
        }

        def fake_merge(*args, status_callback):
            self.assertEqual(len(args), 48)
            self.assertEqual(args[:22], tuple(range(22)))
            self.assertEqual(args[-2:], ("", False))
            status_callback("Loading Model B...")
            status_callback("Merging tensors: 5/10")
            return "Merged model saved in test.safetensors", "test"

        namespace["smergegen"] = fake_merge
        namespace["_merge_lock"] = threading.Lock()
        merge_only = load_function(ROOT / "GenParamGetter.py", "merge_only", namespace)
        updates = list(merge_only(*range(22)))
        self.assertTrue(any("Loading Model B" in status for _, _, status in updates))
        self.assertIn("Saved and loaded successfully", updates[-1][2])
        self.assertEqual(updates[-1][1], "test")


if __name__ == "__main__":
    unittest.main()
