import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "summarize", Path(__file__).resolve().parents[1] / "scripts/analysis/summarize.py"
)
summarize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarize)


def write(tmp_path, experiment, dataset, seed, er):
    name = f"{experiment}_{dataset}_s{seed}.json"
    (tmp_path / name).write_text(
        json.dumps(
            {
                "experiment": experiment,
                "dataset": dataset,
                "seed": seed,
                "test": {"error_rate": er},
            }
        )
    )


def test_folds_are_averaged_within_a_seed_then_across_seeds(tmp_path):
    for seed, (a, b) in {1: (0.6, 0.8), 2: (0.5, 0.7)}.items():
        write(tmp_path, "m", "tut_fold1", seed, a)
        write(tmp_path, "m", "tut_fold2", seed, b)
    table = summarize.load(tmp_path, "error_rate")
    assert set(table) == {("m", "tut")}
    xs = summarize.per_seed(table[("m", "tut")])
    assert xs == pytest.approx([0.7, 0.6])
    mean, std = summarize.mean_std(xs)
    assert mean == pytest.approx(0.65) and std == pytest.approx(0.0707107, abs=1e-6)


def test_compare_flags_indistinguishable_when_gap_is_inside_std():
    close = summarize.compare([0.70, 0.74, 0.72], [0.71, 0.75, 0.73])
    far = summarize.compare([0.70, 0.70, 0.70001], [0.80, 0.80, 0.80001])
    assert close.endswith("indistinguishable")
    assert far.endswith("separated")


def test_files_without_the_metric_are_ignored(tmp_path):
    (tmp_path / "junk.json").write_text(json.dumps({"hello": 1}))
    write(tmp_path, "m", "d", 1, 0.5)
    assert list(summarize.load(tmp_path, "error_rate")) == [("m", "d")]
    assert summarize.load(tmp_path, "f_measure") == {}
