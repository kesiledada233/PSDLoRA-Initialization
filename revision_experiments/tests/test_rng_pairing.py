from __future__ import annotations

import random
import unittest

import numpy as np
import torch

from revision_experiments.scripts.rng import derive_init_seed, seed_initialization, seed_training, training_probe


class RngPairingTests(unittest.TestCase):
    def test_method_initialization_seeds_are_stable_and_distinct(self):
        a = derive_init_seed(1107, "iid_matched")
        b = derive_init_seed(1107, "powerlaw_global_a06")
        self.assertEqual(a, derive_init_seed(1107, "iid_matched"))
        self.assertNotEqual(a, b)

    def test_training_stream_is_reset_after_method_specific_initialization(self):
        probes = []
        for method in ("iid_matched", "powerlaw_global_a06"):
            seed_initialization(derive_init_seed(1107, method))
            random.random(); np.random.random(); torch.rand(8)
            seed_training(1107)
            probes.append(training_probe())
        self.assertEqual(probes[0], probes[1])


if __name__ == "__main__":
    unittest.main()
