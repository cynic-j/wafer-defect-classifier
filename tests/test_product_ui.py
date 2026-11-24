"""Product presentation regressions; model tests remain in test_competition_demo."""
import csv

import numpy as np

from wafer.config import WaferConfig
from wafer.demo import DemoService, _build_markdown
from wafer.demo_ui import (
    ProductUI, SINGLE_EMPTY, report_cards, experiment_markdown,
    about_markdown, public_error, _confusion_display,
)
from wafer.data import CLASS_NAMES


class ServiceStub:
    def __init__(self):
        self.single_calls = 0

    def single(self, path):
        self.single_calls += 1
        arr = np.ones((4, 4, 3), dtype=np.uint8)
        return arr, arr, arr, "实际回调输出"

    def batch(self, paths, progress=None):
        if progress:
            progress((2, 2), desc="已处理 2/2 张 · 成功 1 · 失败 1")
        return [["one.png", "中心型（Center）", 0.8, "成功", ""],
                ["bad.png", "", None, "拒绝/失败", "/home/private/file"]], [["中心型（Center）", 1]], "export.csv", "完成 1/2 张"


def test_upload_preview_never_calls_inference(tmp_path):
    from PIL import Image
    path = tmp_path / "wafer.png"
    arr = np.full((20, 20), 40, dtype=np.uint8)
    arr[2:-2, 2:-2] = 160
    Image.fromarray(arr).save(path)
    service = ServiceStub()
    ui = ProductUI(service)
    preview, original, heat, overlay, md = ui.preview(path)
    assert service.single_calls == 0
    assert preview.shape == original.shape == (20, 20, 3)
    assert heat is overlay is None and "已就绪" in md
    assert ui.preview(None) == (None, None, None, None, SINGLE_EMPTY)
    ui.single(path)
    assert service.single_calls == 1


def test_rejected_preview_and_public_errors(tmp_path):
    ui = ProductUI(ServiceStub())
    output = ui.preview(tmp_path / "missing.png")
    assert output[:4] == (None, None, None, None)
    assert str(tmp_path) not in output[-1]
    for value in ("Permission denied /home/darksky/best.pt", "CUDA error", "C:\\private\\best.pt"):
        public = public_error(value)
        assert "best.pt" not in public and "CUDA" not in public and "/home" not in public


def test_batch_presentation_counts_progress_and_export():
    ui = ProductUI(ServiceStub())
    reports = []
    cards, rows, counts, export, status = ui.batch(["one", "two"], lambda *args, **kw: reports.append((args, kw)))
    assert len(rows[0]) == 4 and rows[0][2] == "80.00%" and rows[1][2] == "—"
    assert "已处理" in cards and "失败" in cards and "<b>2</b>" in cards
    assert "<b>1</b>" in counts and "export.csv" == export
    assert "/home" not in str(rows)
    assert reports[0][0][0] == (2, 2)
    assert "101" in ui.upload_batch(["file"] * 101)[0]
    assert "最多 100" in ui.upload_batch(["file"] * 101)[-1]
    assert ui.upload_batch([])[3] is None


def test_report_source_and_split_labels_without_paths(tmp_path, monkeypatch):
    import wafer.demo_ui as module
    monkeypatch.setattr(module, "EVIDENCE_ROOT", tmp_path / "no-evidence")
    output = tmp_path / "baseline_3090_gpu0_seed42"
    output.mkdir()
    cfg = WaferConfig(device="cpu", output_dir=output)
    assert "暂未提供" in report_cards(cfg)
    (output / "training_summary.txt").write_text("Done. Best val macro-F1: 0.7", encoding="utf-8")
    with (output / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1_score", "support"])
        for name in CLASS_NAMES:
            writer.writerow([name, 0.6, 0.5, 0.55, 2])
        writer.writerow(["macro_avg", 0.6, 0.5, 0.55, ""])
    report = experiment_markdown(cfg)
    cards = report_cards(cfg)
    assert "0.7000" in report and "0.5500" in report
    assert "验证" in cards and "独立测试" in cards and "18" in cards
    assert str(tmp_path) not in report + cards
    ablation_line = next(line for line in report.splitlines() if line.startswith("| Ablation-01"))
    assert ablation_line.count("暂未提供") == 3
    assert _confusion_display(cfg, tmp_path)[0] is None


def test_no_developer_details_in_public_content():
    cfg = WaferConfig(device="cpu")
    markdown = _build_markdown(0, np.full(9, 1 / 9), 1)
    assert "置信度" in markdown and "未校准" in markdown and "T=" not in markdown
    about = about_markdown()
    assert "不是像素级缺陷分割" in about
    for token in ("checkpoint", "CUDA", "num_workers", "best.pt", "/home/", "batch_size"):
        assert token not in markdown + about


def test_service_batch_progress_without_changing_decisions(tmp_path, monkeypatch):
    import wafer.demo as demo
    cfg = WaferConfig(device="cpu")
    service = DemoService(cfg, object(), None, 1.0, {}, tmp_path)
    arr = np.full((4, 4, 3), 160, dtype=np.uint8)
    monkeypatch.setattr(demo, "_read_png", lambda path: arr if path == "valid" else (_ for _ in ()).throw(ValueError("bad PNG")))
    probs = np.zeros(9)
    probs[0] = 1
    monkeypatch.setattr(demo, "_predict_probabilities", lambda *args: (0, probs))
    updates = []
    rows, counts, export, _ = service.batch(["valid", "bad"], lambda value, **kwargs: updates.append((value, kwargs["desc"])))
    assert [item[0] for item in updates] == [(0, 2), (1, 2), (2, 2)]
    assert "成功 1" in updates[-1][1] and "失败 1" in updates[-1][1]
    assert rows[0][1] == "中心型（Center）" and sum(row[1] for row in counts) == 1
    assert export.endswith(".csv")
