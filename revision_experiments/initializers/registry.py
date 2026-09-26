"""Canonical names and immutable descriptions for initializer variants."""

from __future__ import annotations

METHOD_REGISTRY = {
    "peft_default": {
        "alpha": None,
        "family": "library_default",
        "scale_match": None,
        "description": "Installed PEFT reset_lora_parameters path; B must remain zero.",
    },
    "iid_matched": {
        "alpha": None,
        "family": "iid_normal",
        "scale_match": "powerlaw_global_a06",
        "description": "I.i.d. normal control matched per matrix to the a=0.6 global variant.",
    },
    "fft_white_a0": {
        "alpha": 0.0,
        "family": "fft_global",
        "scale_match": "powerlaw_global_a06",
        "description": "Flat-amplitude FFT control, scale matched to the a=0.6 variant.",
    },
    "powerlaw_global_a03": {
        "alpha": 0.3,
        "family": "fft_global",
        "scale_match": None,
        "description": "Whole flattened matrix power-law synthesis with a=0.3.",
    },
    "powerlaw_global_a06": {
        "alpha": 0.6,
        "family": "fft_global",
        "scale_match": None,
        "description": "Whole flattened matrix power-law synthesis with a=0.6.",
    },
    "powerlaw_global_a10": {
        "alpha": 1.0,
        "family": "fft_global",
        "scale_match": None,
        "description": "Whole flattened matrix power-law synthesis with a=1.0.",
    },
    "powerlaw_shuffle_a06": {
        "alpha": 0.6,
        "family": "permuted_global",
        "scale_match": "exact_values",
        "description": "Random permutation of the exact a=0.6 global values.",
    },
    "powerlaw_row_a06": {
        "alpha": 0.6,
        "family": "fft_row",
        "scale_match": "powerlaw_global_a06",
        "description": "Independent power-law synthesis for each row.",
    },
    "powerlaw_col_a06": {
        "alpha": 0.6,
        "family": "fft_column",
        "scale_match": "powerlaw_global_a06",
        "description": "Independent power-law synthesis for each column.",
    },
}


def registered_methods() -> tuple[str, ...]:
    return tuple(METHOD_REGISTRY)


def require_registered(method: str) -> None:
    if method not in METHOD_REGISTRY:
        allowed = ", ".join(registered_methods())
        raise ValueError(f"Unknown initialization method {method!r}; allowed: {allowed}")
