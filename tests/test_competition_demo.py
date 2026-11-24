"""Regression tests use temporary random weights, never experimental assets."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wafer.config import WaferConfig
from wafer.data import CLASS_NAMES, CLASS_TO_IDX
from wafer.demo import (
    DemoService, _load_assets, _read_png, _run_inference, _validate_image,
    _build_markdown, _predict_probabilities, _demo_config,
)
from wafer.demo_ui import load_class_metrics, experiment_markdown, experiment_record, evidence_path
from wafer.model import build_model, CBAM


def wafer_png(path):
    from PIL import Image
    arr = np.full((32, 32), 40, dtype=np.uint8)
    arr[3:-3, 3:-3] = 160
    arr[12:18, 12:18] = 255
    Image.fromarray(arr).save(path)
    return arr


@pytest.fixture
def assets(tmp_path):
    cfg = WaferConfig(device="cpu", output_dir=tmp_path, input_size=64,
                      arch="resnet18", cbam=True, tta=True, pretrained=False)
    model = build_model(cfg)
    torch.save({"model_state_dict": model.state_dict(), "class_to_idx": CLASS_TO_IDX,
                "cfg": cfg.to_dict(), "val_macro_f1": 0.0}, tmp_path / "best.pt")
    # Deliberately conflicting UI config must be restored from checkpoint.
    cfg.arch, cfg.cbam, cfg.input_size, cfg.pretrained = "resnet50", False, 224, True
    loaded, layer, temperature, thresholds = _load_assets(cfg)
    return cfg, loaded, layer, temperature, thresholds


def test_checkpoint_restore_and_real_tta_gradcam(assets, tmp_path):
    cfg, model, layer, temperature, thresholds = assets
    assert cfg.arch == "resnet18" and cfg.cbam and cfg.input_size == 64
    assert not cfg.pretrained  # No ImageNet downloads when loading saved weights.
    assert isinstance(model.layer4[-1], CBAM) and layer is model.layer4[0]
    assert temperature == 1 and thresholds == {} and not model._wafer_calibrated
    arr = wafer_png(tmp_path / "wafer.png")
    tensor, heat, pred, probs = _run_inference(arr, model, layer, temperature, thresholds, cfg)
    assert tensor.shape == (3, 64, 64) and heat.shape == (64, 64)
    assert np.isfinite(heat).all() and heat.min() >= 0 and heat.max() <= 1.00001
    assert probs.shape == (9,) and np.isclose(probs.sum(), 1)
    assert pred == probs.argmax()
    batch_pred, batch_probs = _predict_probabilities(arr, model, temperature, thresholds, cfg)
    assert pred == batch_pred
    np.testing.assert_allclose(probs, batch_probs, atol=1e-6)
    assert len(layer._forward_hooks) == len(layer._backward_hooks) == 0


@pytest.mark.parametrize("arr", [None, np.zeros((0, 2)), np.zeros((2, 2, 4)),
    np.full((4, 4), np.nan), np.ones((4, 4)), np.full((4, 4), 40),
    np.full((4, 4, 3), [0, 225, 255]), np.full((4, 4), 300)])
def test_invalid_inputs_rejected(arr):
    with pytest.raises(ValueError):
        _validate_image(arr)


def test_png_validation(tmp_path):
    from PIL import Image
    valid = tmp_path / "wafer.png"
    wafer_png(valid)
    assert _read_png(valid).shape == (32, 32, 3)
    fake = tmp_path / "not-really-png.png"
    Image.new("RGB", (32, 32), (160, 160, 160)).save(fake, format="JPEG")
    with pytest.raises(ValueError, match="PNG"):
        _read_png(fake)
    alpha = tmp_path / "alpha.png"
    Image.new("RGBA", (32, 32), (160, 160, 160, 128)).save(alpha)
    with pytest.raises(ValueError):
        _read_png(alpha)


def test_callbacks_export_and_failure_clear(assets, tmp_path):
    service = DemoService(*assets, tmp_path)
    image = tmp_path / "=wafer.png"
    wafer_png(image)
    original, heat, overlay, md = service.single(image)
    assert original.shape == (32, 32, 3) and heat.shape == overlay.shape == (64, 64, 3)
    assert "未校准" in md and "校准置信度" not in md
    assert service.single(None)[:3] == (None, None, None)
    rows, counts, export, status = service.batch([image, tmp_path / "missing.png"])
    assert rows[0][3] == "成功" and rows[1][3] == "拒绝/失败"
    assert sum(r[1] for r in counts) == 1
    with Path(export).open(encoding="utf-8-sig") as handle:
        saved = list(csv.reader(handle))
    assert saved[1][0].startswith("'=") and saved[1][-3:] == ["未校准", "TTA×8", "argmax"]
    assert "1/2" in status
    assert service.batch([])[2] is None


def test_optional_assets_and_class_mapping(assets, tmp_path):
    cfg = assets[0]
    (tmp_path / "temperature.json").write_text(json.dumps({"temperature": 1.2}))
    (tmp_path / "thresholds.json").write_text(json.dumps({"Scratch": 0.7}))
    model, _, temperature, thresholds = _load_assets(cfg)
    assert model._wafer_calibrated and temperature == 1.2 and thresholds["Scratch"] == 0.7
    (tmp_path / "temperature.json").write_text('{"temperature": -1}')
    with pytest.raises(ValueError, match="temperature"):
        _load_assets(cfg)


def test_metrics_missing_real_and_invalid(tmp_path):
    rows, message = load_class_metrics(tmp_path)
    assert len(rows) == 9 and "暂未提供" in message
    path = tmp_path / "per_class_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1_score", "support"])
        for c in CLASS_NAMES:
            writer.writerow([c, 0.5, 0.4, 0.44, 10])
    rows, message = load_class_metrics(tmp_path)
    assert rows[0][1:] == ["0.5000", "0.4000", "0.4400", 10]
    path.write_text("class,precision\nCenter,wrong", encoding="utf-8")
    assert "失败" in load_class_metrics(tmp_path)[1]


def test_record_scope_and_none_wording(tmp_path):
    cfg = WaferConfig(device="cpu", output_dir=tmp_path)
    assert "0.9139" not in experiment_markdown(cfg)
    probs = np.zeros(9)
    probs[-1] = 1
    md = _build_markdown(8, probs, 1.0)
    assert "不代表晶圆质量合格" in md and "0.0031" not in md and "0.9157" not in md


def test_utf8_demo_config_and_preview(tmp_path):
    import argparse
    path = tmp_path / "中文配置.yaml"
    path.write_text("# UTF-8 → 配置\narch: resnet18\ncbam: true\n", encoding="utf-8")
    cfg = _demo_config(argparse.Namespace(config=path, device="cpu", batch_size=32))
    assert cfg.cbam and cfg.batch_size == 32
    service = DemoService(cfg, None, None, 1.0, {}, tmp_path)
    assert not service.ready
    assert "静态预览" in service.single(None)[-1]
    assert service.batch([])[2] is None


def test_evidence_fallback_and_outputs_priority(tmp_path, monkeypatch):
    import wafer.demo_ui as ui
    evidence = tmp_path / "evidence"
    fallback = evidence / "run"
    fallback.mkdir(parents=True)
    (fallback / "evaluate.log").write_text("Macro-F1 : 0.5\nBalanced accuracy : 0.4\n", encoding="utf-8")
    monkeypatch.setattr(ui, "EVIDENCE_ROOT", evidence)
    run = tmp_path / "outputs" / "run"
    assert evidence_path(run, "evaluate.log") == fallback / "evaluate.log"
    assert experiment_record(run)["test"] == 0.5
    run.mkdir(parents=True)
    with (run / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1_score", "support"])
        for c in CLASS_NAMES:
            writer.writerow([c, 0.8, 0.8, 0.8, 2])
        writer.writerow(["macro_avg", 0.8, 0.8, 0.8, ""])
    record = experiment_record(run)
    assert record["test"] == record["balanced"] == 0.8 and record["n"] == 18
    assert load_class_metrics(run)[0][0][1] == "0.8000"
