from __future__ import annotations

import unittest

import numpy as np

from revision_experiments.scripts.metrics import temporal_psd_slope


def colored_noise(length: int, alpha: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frequencies = np.fft.rfftfreq(length)
    amplitude = np.zeros_like(frequencies)
    positive = frequencies > 0
    amplitude[positive] = frequencies[positive] ** (-alpha / 2.0)
    phase = rng.uniform(0, 2 * np.pi, len(frequencies))
    spectrum = amplitude * np.exp(1j * phase)
    if length % 2 == 0:
        spectrum[-1] = spectrum[-1].real
    return np.fft.irfft(spectrum, n=length)


class PsdRecoveryTests(unittest.TestCase):
    def test_recovers_white_slope(self):
        result = temporal_psd_slope(colored_noise(16384, 0.0, 1))
        self.assertAlmostEqual(result["alpha"], 0.0, delta=0.15)

    def test_recovers_one_over_f_slope(self):
        result = temporal_psd_slope(colored_noise(16384, 1.0, 2))
        self.assertAlmostEqual(result["alpha"], 1.0, delta=0.15)
        self.assertEqual(result["frequency_band"], [0.01, 0.08])


if __name__ == "__main__":
    unittest.main()
