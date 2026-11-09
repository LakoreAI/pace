import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import PACEConfig
from src.data import (
    audioset_missing_clips,
    audioset_strong_items,
    desed_jams_items,
    hold_out_files,
    read_manifest,
    stratified_holdout,
)
from src.modules.augment import apply_recipe, spec_augment_batch
from src.modules.loss import BinarySEDLoss, REDEPNLoss, build_loss
from src.modules.model import SeparableBlock, build_model
from src.modules.red import red_scan
from src.pipelines.config import TrainingConfig
from src.sed.postprocess import (
    decode_file_events,
    framewise_file_events,
    red_epn_file_events,
    red_presence_file_events,
)
from src.sed.psds import event_based_metrics, merge_window_scores, psds_metrics
from src.sed.streaming import stream_file_events, latency_stats
from src.sed.targets import encode_framewise, encode_red_epn
from src.utils.efficiency import count_macs, measure_latency


def test_fold_is_part_of_result_identity():
    cfg = TrainingConfig(
        data_root="/data/features/tut2017_fold2",
        dataset="tut2017",
        fold=2,
        seed=7,
    )
    assert cfg.result_filename() == "pace_tut2017_s7_fold2.json"
    no_fold = TrainingConfig(data_root="/data/features/urbansed", dataset="urbansed")
    assert no_fold.result_filename() == "pace_urbansed_s42.json"


def test_training_config_validation():
    with pytest.raises(ValueError):
        TrainingConfig(batch_size=0)
    with pytest.raises(ValueError):
        TrainingConfig(eval_every=0)
    with pytest.raises(ValueError):
        TrainingConfig(val_file=None, val_files=None)
    with pytest.raises(ValueError):
        TrainingConfig(best_mode="up")


def test_manifest_loader_groups_rows_and_filters_split(tmp_path):
    manifest = tmp_path / "split.tsv"
    manifest.write_text(
        "path\tonset\toffset\tlabel\tsplit\n"
        "a.wav\t0.0\t1.0\tdog\ttrain\n"
        "a.wav\t2.0\t3.0\tcat\ttrain\n"
        "b.wav\t0.0\t1.0\tdog\ttest\n"
    )
    items = read_manifest(manifest, tmp_path, split="train")
    assert len(items) == 1
    assert items[0][0] == tmp_path / "a.wav"
    assert items[0][1] == [(0.0, 1.0, "dog"), (2.0, 3.0, "cat")]


def test_audioset_loader_and_missing_report(tmp_path):
    (tmp_path / "abc.wav").write_bytes(b"")
    csv = tmp_path / "strong.csv"
    csv.write_text(
        "YTID,start_time_seconds,end_time_seconds,positive_labels\n"
        "abc,0.0,1.0,/m/dog\n"
        "missing,0.0,1.0,/m/cat\n"
    )
    items = audioset_strong_items(csv, tmp_path)
    assert len(items) == 1 and items[0][0].name == "abc.wav"
    assert audioset_missing_clips(csv, tmp_path) == ["missing"]


def test_desed_jams_loader(tmp_path):
    import soundfile as sf

    jams_dir = tmp_path / "jams"
    audio_dir = tmp_path / "audio"
    jams_dir.mkdir()
    audio_dir.mkdir()
    sf.write(audio_dir / "scene.wav", np.zeros(16000), 16000)
    (jams_dir / "scene.jams").write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "namespace": "scaper",
                        "data": [
                            {"time": 0.5, "duration": 1.0, "value": {"label": "dog"}}
                        ],
                    }
                ]
            }
        )
    )
    items = desed_jams_items(jams_dir, audio_dir)
    assert len(items) == 1 and items[0][1] == [(0.5, 1.5, "dog")]


def test_stratified_holdout_keeps_label_sets_on_both_sides():
    items = [
        (Path(f"{i}.wav"), [(0.0, 1.0, "rare" if i < 2 else "common")])
        for i in range(10)
    ]
    train, val = stratified_holdout(items, 0.2, seed=0)
    assert len(train) + len(val) == 10
    assert any(e[2] == "rare" for _, ev in val for e in ev)
    assert any(e[2] == "common" for _, ev in val for e in ev)


def test_framewise_task_roundtrip_and_loss():
    y = encode_framewise([(0.30, 0.50, "b")], {"a": 0, "b": 1}, 2.56, 9)
    assert y.shape == (9, 2) and y[:, 1].sum() == 1
    cfg = PACEConfig(head="framewise", num_classes=2)
    model = build_model(cfg)
    out = model(torch.randn(2, cfg.n_frames, cfg.n_mels))["pred"]
    assert out.shape == (2, cfg.n_cells, 2)
    loss = build_loss({"type": "bce"}, 2, "framewise")
    assert isinstance(loss, BinarySEDLoss)
    target = torch.zeros(2, cfg.n_cells, 2)
    assert torch.isfinite(loss(out, target))


def test_framewise_decoder_uses_model_resolution():
    pred = np.zeros((9, 2), dtype=np.float32)
    pred[2:5, 0] = 0.9
    events = framewise_file_events(
        [pred], [0.0], ["a", "b"], win_len=2.56, threshold=0.5, merge_gap=0.0
    )
    width = 2.56 / 9
    assert len(events) == 1 and events[0][2] == "a"
    assert events[0][0] == pytest.approx(2 * width, abs=1e-3)
    assert events[0][1] == pytest.approx(5 * width, abs=1e-3)
    dispatched = decode_file_events(
        {"type": "framewise"}, [pred], [0.0], ["a", "b"], 2.56, threshold=0.5
    )
    assert dispatched == events


def test_psds_and_event_metrics_are_finite():
    pred = np.zeros((4, 6), dtype=np.float32)
    pred[:, 0] = 0.9
    pred[:, 3] = 0.9
    timestamps, scores = merge_window_scores(pred[None], [0.0], 4.0, "onepass")
    refs = {"f": [(0.0, 2.0, "a"), (2.0, 3.0, "b")]}
    result = psds_metrics(refs, {"f": (timestamps, scores)}, {"f": 4.0}, ["a", "b"])
    assert np.isfinite(result["psds1"]) and np.isfinite(result["psds2"])
    assert result["classes_evaluated"] == ["a", "b"]
    events = {"f": [(0.0, 2.0, "a"), (2.0, 3.0, "b")]}
    assert event_based_metrics(refs, events, ["a", "b"])["event_f_measure"] == 1.0


def test_efficiency_counts_macs_and_latency():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, 3, padding=1), torch.nn.ReLU(), torch.nn.Flatten()
    )
    sample = torch.zeros(2, 1, 8, 8)
    assert count_macs(model, sample) == 2 * 4 * 8 * 8 * 9
    latency = measure_latency(model, sample, warmup=1, repeats=3)
    assert latency["median_ms"] >= 0 and latency["batch_size"] == 2


def test_spec_augment_min_fill_and_recipe_validation():
    x = torch.full((2, 20, 8), -30.0)
    x[0, 0, 0] = 5.0
    out = spec_augment_batch(
        x,
        freq_param=8,
        time_param=5,
        time_mask_num=1,
        fill="min",
        generator=torch.Generator().manual_seed(0),
    )
    assert out.min() >= -30.0
    with pytest.raises(ValueError):
        apply_recipe(x, torch.zeros(2, 3), {"nope": {}})


def test_hold_out_files_is_deterministic():
    items = [(Path(f"{i}.wav"), []) for i in range(10)]
    assert hold_out_files(items, 0.2, seed=0) == hold_out_files(items, 0.2, seed=0)


def test_red_scan_forms_one_event():
    start = torch.full((1, 5, 1), -10.0)
    end = torch.full((1, 5, 1), -10.0)
    start[0, 1] = 10.0
    end[0, 3] = 10.0
    out = red_scan(start, end)
    assert out["pres"][0, :, 0].tolist() == pytest.approx([0, 1, 1, 0, 0], abs=1e-4)
    assert out["on"][0, 1, 0] == pytest.approx(1.0, abs=1e-4)
    assert out["off"][0, 3, 0] == pytest.approx(1.0, abs=2e-4)


def test_red_epn_targets_and_loss_backward():
    target = encode_red_epn([(0.30, 0.50, "a")], {"a": 0, "b": 1}, 2.56, 9)
    assert target.shape == (9, 10)
    assert target[:, 0].sum() == 1 and target[:, 2].sum() == 1
    assert target[:, 4].sum() == 1 and target[1, 8] > 0
    cfg = PACEConfig(head="red_epn", num_classes=2, epn_hidden=8)
    model = build_model(cfg)
    pred = model(torch.randn(2, cfg.n_frames, cfg.n_mels))["pred"]
    assert set(pred) == {"pres", "on", "off", "start", "end", "don", "doff"}
    y = torch.as_tensor(np.stack([target, target]), dtype=torch.float32)
    loss = REDEPNLoss(2)(pred, y)
    assert torch.isfinite(loss)
    loss.backward()


def test_red_epn_decoder_suppresses_duplicates_and_limits_k():
    preds = {
        "pres": np.zeros((1, 9, 1), dtype=np.float32),
        "don": np.zeros((1, 9, 1), dtype=np.float32),
        "doff": np.zeros((1, 9, 1), dtype=np.float32),
    }
    preds["pres"][0, 2:4, 0] = 0.9
    preds["don"][0, 2:4, 0] = 0.0
    preds["doff"][0, 2:4, 0] = 4.0
    events = red_epn_file_events(
        preds, [0.0], ["a"], 2.56, threshold=0.5, iou_threshold=0.5, k=15
    )
    assert len(events) == 1 and events[0][2] == "a"
    limited = red_epn_file_events(
        preds, [0.0], ["a"], 2.56, threshold=0.5, iou_threshold=0.99, k=1
    )
    assert len(limited) == 1
    dispatched = decode_file_events(
        {"type": "red_epn"}, preds, [0.0], ["a"], 2.56, threshold=0.5
    )
    assert dispatched == events


def test_causal_block_does_not_use_future_frames():
    cfg = PACEConfig(causal=True, num_classes=1, dropout=0.0)
    causal = SeparableBlock(1, 1, 3, 1, cfg).eval()
    noncausal_cfg = PACEConfig(causal=False, num_classes=1, dropout=0.0)
    noncausal = SeparableBlock(1, 1, 3, 1, noncausal_cfg).eval()
    with torch.no_grad():
        for block in (causal, noncausal):
            block.depthwise.weight.fill_(1.0)
            block.pointwise.weight.fill_(1.0)
    x = torch.randn(1, 1, 6, 1)
    perturbed = x.clone()
    perturbed[:, :, 1, :] += 5.0
    with torch.no_grad():
        assert torch.allclose(causal(x)[:, :, 0], causal(perturbed)[:, :, 0])
        assert not torch.allclose(noncausal(x)[:, :, 0], noncausal(perturbed)[:, :, 0])


def test_causal_backbone_keeps_output_grid():
    for head in ("onepass", "framewise", "red_epn"):
        cfg = PACEConfig(num_classes=2, causal=True, head=head, epn_hidden=8)
        model = build_model(cfg)
        out = model(torch.randn(1, cfg.n_frames, cfg.n_mels))["pred"]
        if isinstance(out, dict):
            assert out["pres"].shape[1] == cfg.n_cells
        else:
            assert out.shape[1] == cfg.n_cells


def test_streaming_commit_latency_and_force():
    preds = {
        "pres": np.zeros((1, 9, 1), dtype=np.float32),
        "don": np.zeros((1, 9, 1), dtype=np.float32),
        "doff": np.zeros((1, 9, 1), dtype=np.float32),
    }
    preds["pres"][0, 2, 0] = 0.9
    preds["doff"][0, 2, 0] = 2.0
    result = stream_file_events(
        preds, [0.0], ["a"], 2.0, "red_epn", {"threshold": 0.5}, lookahead=0.5
    )
    assert len(result.events) == 1 and result.forced_at_end == 0
    assert result.commit_times[0] == pytest.approx(2.0)
    stats = latency_stats(result.events, result.commit_times)
    assert stats["mean_delay_after_offset"] == pytest.approx(2.0 - result.events[0][1])

    late = preds.copy()
    late["doff"] = np.full_like(preds["doff"], 8.0)
    forced = stream_file_events(
        late, [0.0], ["a"], 2.0, "red_epn", {"threshold": 0.5}, lookahead=0.5
    )
    assert forced.forced_at_end == 1


def test_streaming_deduplicates_repeated_windows():
    preds = {
        "pres": np.zeros((2, 9, 1), dtype=np.float32),
        "don": np.zeros((2, 9, 1), dtype=np.float32),
        "doff": np.zeros((2, 9, 1), dtype=np.float32),
    }
    preds["pres"][:, 2, 0] = 0.9
    preds["doff"][:, 2, 0] = 2.0
    result = stream_file_events(
        preds, [0.0, 0.0], ["a"], 2.0, "red_epn", {"threshold": 0.5}, lookahead=0.0
    )
    assert len(result.events) == 1


def test_red_presence_decoder_uses_presence_runs():
    preds = {"pres": np.zeros((1, 9, 1), dtype=np.float32)}
    preds["pres"][0, 2:5, 0] = 0.9
    events = red_presence_file_events(
        preds, [0.0], ["a"], 2.56, threshold=0.5, median_lengths=3
    )
    assert len(events) == 1 and events[0][2] == "a"
    dispatched = decode_file_events(
        {"type": "red_presence"}, preds, [0.0], ["a"], 2.56, threshold=0.5
    )
    assert dispatched == events


def test_compare_does_not_collapse_fold_tags(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "compare",
        Path(__file__).resolve().parents[1] / "scripts/analysis/compare_benchmarks.py",
    )
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    for fold, value in ((1, 0.6), (2, 0.8)):
        (tmp_path / f"a_fold{fold}.json").write_text(
            json.dumps(
                {
                    "experiment": "onepass",
                    "dataset": f"tut2017_fold{fold}",
                    "seed": 1,
                    "test": {"error_rate": value},
                }
            )
        )
    rows = compare.load_results(tmp_path)
    groups = compare.ours_for(
        rows, "tut2017", {"aliases": ["tut2017"]}, "error_rate", 1.0
    )
    assert groups["onepass"]["n_runs"] == 2
    assert groups["onepass"]["mean_std"][0] == pytest.approx(0.7)
