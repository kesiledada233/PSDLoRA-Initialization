from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from revision_experiments.scripts.gradient_logger import GradientLogger


class Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.ones(2, 4))
        self.lora_B = torch.nn.Parameter(torch.zeros(4, 2))


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = Adapter()


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer(), Layer(), Layer(), Layer()])


class GradientLoggerTests(unittest.TestCase):
    def test_fixed_coordinate_files_cover_first_middle_last(self):
        model = Toy()
        for parameter in model.parameters():
            parameter.grad = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
        with tempfile.TemporaryDirectory() as directory:
            logger = GradientLogger(model, Path(directory) / "gradients", 2, 3, 11)
            logger.record(0); logger.record(1); logger.record(2)
            manifest = json.loads((Path(directory) / "gradients/manifest.json").read_text())
            self.assertEqual(manifest["selected_layers"], [0, 2, 3])
            for entry in manifest["parameters"]:
                values = np.load(Path(directory) / "gradients" / entry["file"])
                self.assertEqual(values.shape, (3, 3))
                self.assertFalse(np.isnan(values).any())


if __name__ == "__main__":
    unittest.main()
