"""Illustrations include errors, and plots reject changed class ordering."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("make_plots", ROOT / "scripts/make_plots.py")
assert SPEC and SPEC.loader
plots = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plots)


def test_selection_contains_correct_and_incorrect_cases_for_every_class() -> None:
    target = np.repeat(np.arange(3), 3)
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.9, 0.05, 0.05],
            [0.01, 0.98, 0.01],
            [0.1, 0.8, 0.1],
            [0.05, 0.9, 0.05],
            [0.01, 0.01, 0.98],
            [0.1, 0.1, 0.8],
            [0.05, 0.05, 0.9],
            [0.98, 0.01, 0.01],
        ]
    )
    chosen = plots._select_sample_indices(target, probabilities)
    assert len(chosen) == len(set(chosen)) == 6
    for label in range(3):
        selected = [i for i in chosen if target[i] == label]
        assert len(selected) == 2
        assert sum(probabilities[i].argmax() == label for i in selected) == 1


def test_plot_gate_rejects_label_order_drift() -> None:
    with np.load(ROOT / "reports/replay-native-2026-10-15.npz", allow_pickle=False) as data:
        target, probabilities = data["target"], data["probabilities"]
    plots._cross_check(target, probabilities)
    with pytest.raises(SystemExit, match="diverge"):
        plots._cross_check(target, probabilities[:, ::-1])
