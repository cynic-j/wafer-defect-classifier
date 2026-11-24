"""Competition UI and explicitly sourced experiment records; no model algorithms."""
from __future__ import annotations

import csv
import inspect
import re
import shutil
import html
from pathlib import Path

from wafer.data import CLASS_NAMES
from wafer.config import REPO_ROOT

TITLE = "晶圆缺陷智能检测与分析系统"
BASELINE_DIR = "baseline_3090_gpu0_seed42"
EVIDENCE_ROOT = REPO_ROOT / "docs" / "wafer_evidence_20261124"
LABELS = {
    "Center": "中心型", "Donut": "环带型", "Edge-Loc": "边缘局部型",
    "Edge-Ring": "边缘环型", "Loc": "局部型", "Near-full": "近全覆盖型",
    "Random": "随机型", "Scratch": "划痕型", "none": "未识别到已知空间模式",
}

CSS = """
:root {color-scheme:light;}
body {margin:0; background:#f4f7fb!important;}
.gradio-container {max-width:1360px!important; padding:24px 32px!important;
 font-family:"Microsoft YaHei","Noto Sans SC","PingFang SC",sans-serif!important; color:#182e48;
 --body-background-fill:#f4f7fb; --body-text-color:#182e48; --body-text-color-subdued:#62758b;
 --block-background-fill:#fff; --block-border-color:#dde5ef; --block-radius:12px;
 --block-label-background-fill:#fff; --block-label-text-color:#62758b; --block-title-text-color:#182e48;
 --input-background-fill:#fff; --input-text-color:#182e48; --background-fill-primary:#fff;
 --background-fill-secondary:#f1f5fa; --border-color-primary:#dde5ef; --border-color-accent:#1769b5;
 --button-primary-background-fill:#1769b5; --button-primary-text-color:#fff;
 --button-secondary-background-fill:#fff; --button-secondary-text-color:#244568;
 --button-primary-background-fill-hover:#10558f; --button-secondary-border-color:#d3dfed;
 --shadow-drop:0 2px 8px #14365a08;}
.gradio-container footer {display:none!important;}
#brand-header {display:flex; align-items:center; justify-content:space-between; gap:16px; padding:6px 0 22px;}
.brand {display:flex; align-items:center; gap:13px;}
.brand-mark {width:48px;height:48px; flex-shrink:0;}
.brand-name {font-size:21px; font-weight:700; letter-spacing:.03em; color:#102e4f;}
.brand-subtitle {font-size:11px; letter-spacing:.08em; margin-top:4px; color:#6d7f92;}
.brand-tag {color:#207c85; border:1px solid #cbe5e4; background:#edf8f6; padding:7px 12px; border-radius:6px; font-size:13px;}
#product-tabs {border:0!important; background:transparent!important;}
#product-tabs > .tab-nav, #product-tabs > [role=tablist] {background:#fff; border:1px solid #dde5ef!important;
 border-radius:10px; padding:6px!important; gap:6px; margin-bottom:24px; overflow-x:auto; display:flex; flex-wrap:nowrap;
 position:sticky; top:0; z-index:15; box-shadow:0 2px 8px #15375905;}
#product-tabs [role=tab] {border:0!important; background:transparent; border-radius:6px; padding:12px 24px;
 color:#5a6e84; font-size:15px; font-weight:600; white-space:nowrap;}
#product-tabs [role=tab][aria-selected=true] {background:#102e4f!important; color:#fff!important;}
#product-tabs .tabitem {padding:0!important; border:0!important; background:transparent!important;}
.gradio-container .prose {font-size:15px; line-height:1.8; color:#40556d;}
.gradio-container .prose h2 {color:#102e4f; font-size:23px; margin:0 0 14px;}
.gradio-container .prose h3 {color:#244568; font-size:18px;}
.gradio-container button {font-family:inherit; border-radius:7px!important; min-height:42px;}
.gradio-container .primary {background:#1769b5!important; border-color:#1769b5!important; color:#fff!important;}
.gradio-container button:disabled {opacity:.5;cursor:not-allowed;}
#hero-layout {background:#102e4f!important; border-radius:14px; padding:32px!important; margin-bottom:24px; gap:20px;}
#hero-layout .form, #hero-layout .block {background:transparent!important; border:0!important;}
.hero-copy {padding:18px 0 10px; color:#fff;}
.eyebrow {font-size:12px; letter-spacing:.17em; color:#7bd4d5; font-weight:600; margin-bottom:18px;}
.hero-copy h1 {font-size:36px; line-height:1.5; letter-spacing:.02em; font-weight:700; color:#fff; margin:0 0 16px;}
.hero-copy p {font-size:17px; line-height:1.9; color:#bed0e2; margin:0 0 14px; max-width:540px;}
#hero-actions {max-width:430px; gap:12px;}
#hero-actions button {font-weight:600; min-height:48px;}
#hero-layout button.primary {background:#1769b5!important; border:1px solid #1769b5!important; color:#fff!important;}
#hero-actions .secondary {background:#193e62!important; border:1px solid #6384a3!important; color:#fff!important;}
.wafer-visual {display:flex; align-items:center; justify-content:center; flex-direction:column; height:280px;}
.wafer-visual svg {height:230px; width:230px;}
.wafer-visual small {font-size:12px; color:#8faac4; margin-top:10px;letter-spacing:.08em;}
.feature-grid {display:grid; grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin:0 0 28px;}
.feature-card {background:#fff; border:1px solid #dde5ef;border-radius:10px;padding:24px;}
.feature-icon {width:40px; height:40px; border-radius:9px; background:#eaf3fc;color:#1769b5;display:flex;align-items:center;justify-content:center;font-size:19px;font-weight:700;}
.feature-card h3 {font-size:18px;color:#173857;margin:16px 0 8px;}
.feature-card p {font-size:14px;line-height:1.9;color:#697d92;margin:0;}
.section-heading {font-size:22px;color:#123553;margin:28px 0 8px;font-weight:700;}
.section-caption {font-size:14px;color:#718398;margin-bottom:18px;line-height:1.8;}
.defect-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;}
.defect-chip {display:flex;align-items:center;gap:10px;background:#fff;border:1px solid #dde5ef;padding:14px 18px;border-radius:8px;}
.defect-chip i {width:7px;height:7px;background:#279f9e;border-radius:50%;flex-shrink:0;}
.defect-chip strong {font-size:14px;color:#25445f;}
.defect-chip span {font-size:12px;color:#8190a1;margin-left:auto;}
.page-heading {margin:4px 0 22px;}
.page-heading h2 {font-size:26px;color:#102e4f;margin:0 0 9px;}
.page-heading p {font-size:14px;line-height:1.8;color:#718398;margin:0;}
.workspace-card {border:1px solid #dde5ef!important;background:#fff!important;border-radius:12px!important;padding:20px!important;}
#single-result {min-height:370px;background:#fff!important;padding:24px!important;border:1px solid #dde5ef!important;border-radius:12px;}
#single-result .prose h2 {font-size:26px;margin-top:0;}
#single-result .prose h3:first-of-type {font-size:27px;color:#1769b5;}
#single-result .prose table {font-size:14px;}
#single-result .prose {font-size:14px;}
.result-empty {display:flex;min-height:330px;align-items:center;justify-content:center;flex-direction:column;color:#8c9aaa;text-align:center;}
.result-empty strong {font-size:18px;color:#4b657f;margin:14px 0 8px;}
.result-empty p {font-size:14px;line-height:1.9;max-width:300px;}
.notice {padding:12px 16px;border:1px solid #d7e9ed;border-left:3px solid #299c9e;border-radius:6px;background:#f0f8f9;color:#4d697d;font-size:13px;line-height:1.8;margin:12px 0;}
.notice.error {background:#fff5f1;border-color:#eed7ce;border-left-color:#b76a48;color:#88543c;}
.metric-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:18px 0 24px;}
.metric-card {background:#fff;border:1px solid #dde5ef;border-top:3px solid #209caa;padding:20px;border-radius:9px;}
.metric-card span {font-size:13px;color:#60768c;}
.metric-card strong {display:block;font-size:30px;color:#153a59;margin:9px 0;}
.metric-card small {font-size:12px;color:#8a99a9;}
.batch-counters {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:8px 0 12px;}
.batch-counter {border:1px solid #dde5ef;border-radius:8px;padding:16px 20px;background:#fff;display:flex;justify-content:space-between;align-items:center;color:#63788e;font-size:14px;}
.batch-counter b {font-size:24px;color:#174d78;}
.batch-counter.failed b {color:#a96540;}
.count-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;}
.count-item {background:#fff;border:1px solid #e0e7ef;border-radius:7px;padding:11px 15px;display:flex;justify-content:space-between;align-items:center;font-size:13px;color:#586f86;}
.count-item b {color:#217d87;font-size:18px;}
#batch-results {min-height:160px;}
.product-footer {font-size:12px;color:#8b9aad;text-align:center;padding:24px 0 6px;}
.about-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;}
.about-card {padding:24px;background:#fff;border:1px solid #dde5ef;border-radius:10px;color:#62788d;font-size:14px;line-height:1.9;}
.about-card h3 {color:#173857;font-size:18px;margin:0 0 10px;}
@media(min-width:1600px) {.gradio-container {padding-top:32px!important;}}
@media(max-width:1100px) {.gradio-container {padding:18px 22px!important;} .hero-copy h1{font-size:30px;} #hero-layout{padding:24px!important;} #product-tabs [role=tab]{padding:11px 17px;}}
@media(max-width:760px) {.gradio-container{padding:12px!important;} .brand-name{font-size:17px;} .brand-tag{display:none;}
 .brand-subtitle{font-size:9px;} #brand-header{padding-bottom:14px;} #hero-layout{padding:20px!important;}
 .hero-copy h1{font-size:27px;} .hero-copy p{font-size:15px;} .wafer-visual{height:190px;} .wafer-visual svg{height:155px;width:155px;}
 .feature-grid,.about-grid{grid-template-columns:1fr;} .defect-grid{grid-template-columns:repeat(2,minmax(0,1fr));}
 .metric-grid{grid-template-columns:repeat(2,minmax(0,1fr));}.count-grid{grid-template-columns:repeat(2,minmax(0,1fr));}
 .batch-counter{padding:12px;}.batch-counter b{font-size:20px;} .workspace-card{padding:14px!important;} #single-result{min-height:300px;}}
@media(max-width:420px) {.brand-name{font-size:14px;}.brand-mark{width:36px;height:36px;}.defect-grid{grid-template-columns:1fr;}
 .hero-copy h1{font-size:24px;}.metric-card{padding:14px;}.metric-card strong{font-size:24px;}}
"""


def class_label(name):
    return f"{LABELS.get(name, name)}（{name}）"


def _record_context(cfg):
    # Never attach Baseline-01 numbers to a different selected model directory.
    return Path(cfg.output_dir).name == BASELINE_DIR


def evidence_path(output_dir, filename):
    """Prefer deployment artifacts, fall back only to the same experiment's copy."""
    primary = Path(output_dir) / filename
    if primary.is_file():
        return primary
    fallback = EVIDENCE_ROOT / Path(output_dir).name / filename
    return fallback if fallback.is_file() else None


def experiment_record(output_dir):
    """Parse real logs/CSV; never fill a missing metric with a built-in score."""
    record = {"val": None, "test": None, "balanced": None, "n": None, "sources": []}
    ranks = {key: 99 for key in ("val", "test", "balanced", "n")}
    for filename in ("training_summary.txt", "evaluate.log"):
        path = evidence_path(output_dir, filename)
        if path is None:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue
        record["sources"].append(str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path))
        rank = 0 if path.parent == Path(output_dir) else 1
        patterns = {"val": r"(?:Best val macro-F1:|val macro-F1)\s*([0-9.]+)",
                    "test": r"^\s*Macro-F1\s*:\s*([0-9.]+)",
                    "balanced": r"^\s*Balanced accuracy\s*:\s*([0-9.]+)",
                    "n": r"test:\s*([\d,]+)"}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches and rank <= ranks[key]:
                try:
                    value = float(matches[-1].replace(",", ""))
                    if key == "n" and value >= 0:
                        record[key] = int(value)
                        ranks[key] = rank
                    elif key != "n" and 0 <= value <= 1:
                        record[key] = value
                        ranks[key] = rank
                except ValueError:
                    pass
    # The native CSV contains macro F1, macro recall (= balanced accuracy), support.
    path = evidence_path(output_dir, "per_class_metrics.csv")
    if path:
        rank = 0 if path.parent == Path(output_dir) else 1
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            macro = next(r for r in rows if r.get("class") == "macro_avg")
            for key, column in (("test", "f1_score"), ("balanced", "recall")):
                value = float(macro[column])
                if rank <= ranks[key] and 0 <= value <= 1:
                    record[key] = value
                    ranks[key] = rank
            if rank <= ranks["n"]:
                selected = [r for r in rows if r.get("class") in CLASS_NAMES]
                if len(selected) == 9 and len({r["class"] for r in selected}) == 9:
                    record["n"] = sum(int(r["support"]) for r in selected)
            record["sources"].append(str(path))
        except (OSError, ValueError, KeyError, StopIteration, csv.Error):
            pass
    return record


def metric(value):
    return "暂未提供" if value is None else f"{value:.4f}"


def experiment_markdown(cfg):
    if not _record_context(cfg):
        return "当前选择的不是 Baseline-01 目录，不套用 Baseline-01 的实验记录。"
    baseline = experiment_record(cfg.output_dir)
    ablation = experiment_record(Path(cfg.output_dir).parent / "ablation_no_cbam_seed42")
    result = (
        "### Baseline-01 与 CBAM 消融对照\n\n"
        "| 实验 | 模型 | 最佳验证 Macro-F1 | 独立测试 Macro-F1 | Balanced Accuracy |\n|---|---|---:|---:|---:|\n"
        f"| Baseline-01 | ResNet-18 + CBAM + Focal Loss | {metric(baseline['val'])} | {metric(baseline['test'])} | {metric(baseline['balanced'])} |\n"
        f"| Ablation-01 | ResNet-18 + Focal Loss，无 CBAM | {metric(ablation['val'])} | {metric(ablation['test'])} | {metric(ablation['balanced'])} |\n\n"
        f"Baseline-01 测试样本数：**{baseline['n'] if baseline['n'] is not None else '暂未提供'}**。"
        "该实验测试使用八视图增强，尚未进行温度校准。\n\n"
        "上述是已保存的实验记录，不是当前上传图像的准确率，也不是本机重新评估结果。"
        "无 CBAM 组的独立测试指标未提供时，不进行测试性能对比。\n\n"
        "数据来源：已保存的训练记录与独立测试报告。"
    )
    if baseline['val'] is not None and ablation['val'] is not None:
        delta = baseline['val'] - ablation['val']
        result += f"\n\n两组验证集差值为 {delta:.4f}（{delta*100:.2f} 个百分点），仅为单次实验对照，不证明独立测试提升或统计显著性。"
    return result


def load_class_metrics(output_dir):
    """Read only this run's evaluate.py CSV. Missing/invalid values stay unknown."""
    path = evidence_path(output_dir, "per_class_metrics.csv")
    empty = [[class_label(c), "暂未提供", "暂未提供", "暂未提供", "暂未提供"] for c in CLASS_NAMES]
    if path is None:
        return empty, "九类指标暂未提供。"
    try:
        import math
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        selected = [r for r in rows if r.get("class") in CLASS_NAMES]
        if len(selected) != 9 or len({r["class"] for r in selected}) != 9:
            raise ValueError("CSV 必须包含九个不重复的标准类别")
        by_class = {r["class"]: r for r in selected}
        result = []
        for c in CLASS_NAMES:
            r = by_class[c]
            values = [float(r[k]) for k in ("precision", "recall", "f1_score")]
            support = int(r["support"])
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values) or support < 0:
                raise ValueError("指标或样本数量超出有效范围")
            result.append([class_label(c), *[f"{v:.4f}" for v in values], support])
        return result, "数据来源：已保存的九类独立测试报告。"
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        return empty, "九类指标暂未提供，报告文件读取失败。"


def wafer_symbol(large=False):
    """Brand illustration only: no synthetic detection output or experiment data."""
    cells = ''.join(f'<rect x="{x}" y="{y}" width="14" height="14" rx="2"/>'
                    for y in range(30, 215, 18) for x in range(30, 215, 18))
    suffix = "hero" if large else "logo"
    return (f'<svg viewBox="0 0 260 260" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">'
            f'<defs><clipPath id="wafer-{suffix}"><circle cx="130" cy="130" r="104"/></clipPath></defs>'
            '<circle cx="130" cy="130" r="122" fill="#153a5b" stroke="#407191" stroke-width="2"/>'
            f'<g clip-path="url(#wafer-{suffix})" fill="#396381">{cells}</g>'
            '<path d="M130 11V23M130 237V249M11 130H23M237 130H249" stroke="#70ced0" stroke-width="3"/>'
            '<path d="M114 238L130 225L146 238" fill="none" stroke="#70ced0" stroke-width="3"/></svg>')


def brand_markup():
    return ('<header id="brand-header"><div class="brand"><div class="brand-mark">' + wafer_symbol() +
            '</div><div><div class="brand-name">晶圆缺陷智能检测与分析系统</div>'
            '<div class="brand-subtitle">WAFER DEFECT INTELLIGENCE &amp; ANALYSIS PLATFORM</div></div></div>'
            '<span class="brand-tag">晶圆空间模式智能分析</span></header>')


def defect_markup():
    return '<div class="defect-grid">' + ''.join(
        f'<div class="defect-chip"><i></i><strong>{LABELS[c]}</strong><span>{c}</span></div>'
        for c in CLASS_NAMES) + '</div>'


def heading(title, subtitle):
    return f'<div class="page-heading"><h2>{html.escape(title)}</h2><p>{html.escape(subtitle)}</p></div>'


def report_cards(cfg):
    record = experiment_record(cfg.output_dir) if _record_context(cfg) else {}
    cards = [("独立测试 Macro-F1", metric(record.get('test')), "测试集 · 分类综合表现"),
             ("Balanced Accuracy", metric(record.get('balanced')), "测试集 · 类别均衡准确率"),
             ("独立测试样本", str(record['n']) if record.get('n') is not None else "暂未提供", "测试集 · 晶圆图数量"),
             ("最佳验证 Macro-F1", metric(record.get('val')), "验证集 · 模型选择指标")]
    return '<div class="metric-grid">' + ''.join(
        f'<div class="metric-card"><span>{k}</span><strong>{v}</strong><small>{note}</small></div>'
        for k, v, note in cards) + '</div>'


SINGLE_EMPTY = ('<div class="result-empty"><div class="feature-icon">◎</div><strong>等待检测</strong>'
                '<p>上传晶圆图后点击“开始检测”，查看缺陷类别、模型置信度和候选类别。</p></div>')


def batch_counters(total, done=0, failed=0):
    return '<div class="batch-counters">' + ''.join(
        f'<div class="batch-counter {css}"><span>{label}</span><b>{value}</b></div>'
        for label, value, css in (("已上传", total, ""), ("已处理", done, ""), ("失败", failed, "failed"))) + '</div>'


def class_counts(rows=None):
    values = dict(rows or [])
    return '<div class="count-grid">' + ''.join(
        f'<div class="count-item"><span>{LABELS[c]}</span><b>{int(values.get(class_label(c), 0))}</b></div>'
        for c in CLASS_NAMES) + '</div>'


def public_error(message):
    """Show actionable input errors, never expose host paths or stack traces."""
    text = str(message)
    if any(token in text.lower() for token in ("permission", "no such file", "cannot identify", "not found")):
        return "图片读取失败，请重新上传完整的 PNG 文件。"
    if any(token in text for token in ("\\", "/home/", "Traceback", "CUDA", "cuda", "checkpoint", "state_dict")):
        return "检测服务暂时不可用，请重试或联系系统管理员。"
    # Known validation messages contain no user-controlled paths.
    if any(token in text for token in ("灰度", "像素", "PNG", "晶粒", "透明", "图片", "请先", "静态预览")):
        return text.split("**检测未完成：**")[-1].split("\n")[0].strip()
    return "检测未完成，请确认图片格式后重试。"


class ProductUI:
    """Presentation adapters. Upload preview never calls model inference."""
    def __init__(self, service):
        self.service = service

    def preview(self, path):
        if not path:
            return None, None, None, None, SINGLE_EMPTY
        from wafer.demo import _read_png
        try:
            image = _read_png(path)
            return image, image, None, None, "### 图片已就绪\n\n点击“开始检测”获取分析结果。"
        except Exception as exc:
            return None, None, None, None, f"**图片未接受：** {public_error(exc)}"

    def single(self, path):
        result = self.service.single(path)
        if result[0] is None:
            return None, None, None, f"**检测未完成：** {public_error(result[-1])}"
        return result

    def upload_batch(self, paths):
        total = len(paths or [])
        status = f"已上传 {total} 张图片，点击“开始批量检测”。" if total else "等待上传晶圆图。"
        if total > 100:
            status = "单批最多 100 张，请移除多余图片后检测。"
        return batch_counters(total), [], class_counts(), None, status

    def batch(self, paths, progress=None):
        try:
            rows, counts, export, status = self.service.batch(paths, progress=progress)
            done = len(rows)
            failed = sum(row[3] != "成功" for row in rows)
            table = [[row[0], row[1] or "—", f"{row[2]:.2%}" if row[2] is not None else "—", row[3]] for row in rows]
            return batch_counters(len(paths or []), done, failed), table, class_counts(counts), export, status
        except Exception as exc:
            return batch_counters(len(paths or [])), [], class_counts(), None, public_error(exc)


def about_markdown():
    from wafer.demo import _PROCESS_NOTES
    descriptions = "\n".join(f"| {class_label(c)} | {_PROCESS_NOTES[c]} |" for c in CLASS_NAMES)
    return """## 了解晶圆缺陷智能分析

晶圆测试会形成大量晶粒状态记录。本系统面向电子信息与半导体检测研究，
帮助使用者从晶圆空间分布中识别已知缺陷模式，并通过可视化结果支持人工复核。

### 系统能做什么

- **智能识别：** 对单张晶圆状态图给出类别、模型置信度与 Top-3 候选。
- **缺陷可视化：** 并排查看原图、模型关注热力图和叠加图，辅助理解分类依据。
- **批量分析：** 逐张检测、统计类别数量并下载 CSV，记录未通过检查的图片。
- **性能报告：** 查看独立测试结果、九类指标及注意力模块消融对照。

### 三步开始使用

1. 在智能检测页上传 PNG，确认预览符合预期。图片需使用约定灰度：晶圆外部 40、通过晶粒 160、失效晶粒 255。
2. 点击“开始检测”，等待分析完成，查看类别、候选结果和关注区域。
3. 结合原始图和生产背景复核；多张图可使用批量检测并下载结果。

### 九类空间模式

| 类别 | 空间特征 |
|---|---|
""" + descriptions + """

### 技术实现概述

系统基于 ResNet-18 分类网络与 CBAM 注意力模块，利用多视图增强提高推理稳定性，
并使用 Grad-CAM++ 展示模型关注区域。上传预览不运行模型，点击检测后才执行分析。
这些基础算法来自原开源项目；本项目完成中文工作台、输入检查、批量分析、结果导出和实验资料展示整合。

### 数据来源

使用公开 WM-811K 晶圆状态图数据集。输入为晶粒级离散状态图，不是普通照片或 SEM 检测图。
模型性能报告来自已保存的真实实验记录；使用者上传图像的结果与离线实验成绩分别展示。

### 适用范围与局限性

- 适用于九类单标签晶圆空间模式研究与辅助分析，尚不支持混合缺陷识别。
- 图片格式检查不能证明数据来自真实产线；不接受普通照片、SEM 图或带图例的截图。
- “未识别到已知空间模式”不代表晶圆质量合格，未知缺陷仍可能存在。
- 模型置信度为输出概率；未经校准时不能解释为真实正确率。
- 热力图表示分类关注区域，不是像素级缺陷分割，也不是已验证的工艺根因诊断。
- 检测结果需要人工复核，不支持自动放行、判废或设备根因归责。单批最多 100 张。

### 开源许可

基于 [ALEX8642/wafer-defect-classifier](https://github.com/ALEX8642/wafer-defect-classifier)。
遵循 MIT License，保留 Copyright (c) 2026 Alex8642 及原始许可声明。
"""


def _confusion_display(cfg, tmp):
    path = evidence_path(cfg.output_dir, "confusion_matrix.png")
    if path is None:
        return None, "混淆矩阵暂无数据。"
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        display = tmp / "confusion_matrix.png"
        shutil.copyfile(path, display)
        return str(display), "独立测试集混淆矩阵：左侧为样本数量，右侧为按真实类别归一化的比例。"
    except (OSError, ValueError):
        return None, "混淆矩阵暂时无法读取，请联系管理员核对报告。"


def create_ui(gr, cfg, service):
    ui = ProductUI(service)
    kwargs = {"title": TITLE}
    if "css" in inspect.signature(gr.Blocks).parameters:
        kwargs["css"] = CSS
    table, metrics_note = load_class_metrics(cfg.output_dir)
    cm_display, cm_note = _confusion_display(cfg, service.tmp)
    with gr.Blocks(**kwargs) as demo:
        gr.HTML(brand_markup())
        if not service.ready:
            gr.HTML('<div class="notice">当前为静态预览，未加载检测模型。可以浏览产品页面与真实实验报告，检测功能暂不可用。</div>')
        with gr.Tabs(elem_id="product-tabs") as tabs:
            with gr.Tab("系统首页", id="overview"):
                with gr.Row(elem_id="hero-layout"):
                    with gr.Column(scale=3, min_width=300):
                        gr.HTML('<div class="hero-copy"><div class="eyebrow">WAFER INTELLIGENCE</div>'
                                '<h1>晶圆缺陷智能检测<br>与分析系统</h1>'
                                '<p>面向晶圆缺陷识别的智能分析与辅助决策。</p>'
                                '<p>从空间模式识别到关注区域可视化，让每一张晶圆图更易理解。</p></div>')
                        with gr.Row(elem_id="hero-actions"):
                            nav_single = gr.Button("单张检测 →", variant="primary")
                            nav_batch = gr.Button("批量检测 →")
                    with gr.Column(scale=2, min_width=260):
                        gr.HTML('<div class="wafer-visual">' + wafer_symbol(True) + '<small>晶圆结构示意 · 非检测结果</small></div>')
                gr.HTML('<div class="feature-grid">'
                        '<article class="feature-card"><div class="feature-icon">◎</div><h3>智能识别</h3><p>识别已知晶圆缺陷空间模式，提供候选类别和模型置信度，支持人工复核。</p></article>'
                        '<article class="feature-card"><div class="feature-icon">⊞</div><h3>缺陷可视化</h3><p>原图、热力图与叠加图并排呈现，直观理解模型关注区域。</p></article>'
                        '<article class="feature-card"><div class="feature-icon">▤</div><h3>批量分析</h3><p>多张晶圆图连续检测、类别统计与结果导出，让复核流程更清晰。</p></article></div>')
                gr.HTML('<div class="section-heading">支持的空间模式</div><div class="section-caption">覆盖九类已知模式。分类结果用于辅助分析，不替代生产质量判定。</div>' + defect_markup())
            with gr.Tab("智能检测", id="single"):
                gr.HTML(heading("智能检测", "上传晶圆图，查看缺陷类别与模型关注区域。"))
                with gr.Row():
                    with gr.Column(scale=1, min_width=300, elem_classes=["workspace-card"]):
                        upload = gr.File(label="上传晶圆图", file_types=[".png"], type="filepath")
                        preview = gr.Image(label="上传图片预览", interactive=False, height=240)
                        with gr.Row():
                            button = gr.Button("开始检测", variant="primary", interactive=service.ready)
                            clear = gr.Button("清空")
                        gr.Markdown("仅支持约定灰度 PNG：**外部 40 · 通过 160 · 失效 255**。不支持普通照片、SEM 图或带图例截图。")
                        examples = sorted((Path(cfg.output_dir) / "demo_examples").glob("*.png"))
                        if examples:
                            gr.Examples([[str(p)] for p in examples], inputs=[upload], label="选择示例晶圆图")
                    with gr.Column(scale=2, min_width=340):
                        result = gr.Markdown(SINGLE_EMPTY, elem_id="single-result")
                gr.HTML('<div class="section-heading">检测结果可视化</div><div class="section-caption">热力图展示分类关注区域，不是像素级缺陷分割结果。</div>')
                with gr.Row():
                    original = gr.Image(label="原始晶圆图", interactive=False, height=300)
                    heat = gr.Image(label="Grad-CAM++ 热力图", interactive=False, height=300)
                    overlay = gr.Image(label="关注区域叠加图", interactive=False, height=300)
                upload.change(ui.preview, inputs=[upload], outputs=[preview, original, heat, overlay, result], queue=False)
                clear.click(lambda: (None, None, None, None, None, SINGLE_EMPTY),
                            outputs=[upload, preview, original, heat, overlay, result], queue=False)

                def start_single():
                    return gr.Button(interactive=False), gr.File(interactive=False), gr.Button(interactive=False), "### 正在检测…\n\n正在分析缺陷模式并生成关注区域，请稍候。", None, None

                def finish_single():
                    return gr.Button(interactive=service.ready), gr.File(interactive=True), gr.Button(interactive=True)

                def run_single(path, progress=gr.Progress()):
                    progress(0, desc="正在分析晶圆图并生成关注区域…")
                    value = ui.single(path)
                    progress(1, desc="分析已结束")
                    return value

                event = button.click(start_single, outputs=[button, upload, clear, result, heat, overlay], queue=False)
                event = event.then(run_single, inputs=[upload], outputs=[original, heat, overlay, result],
                                   concurrency_id="wafer-model", concurrency_limit=1, show_progress="full")
                event.then(finish_single, outputs=[button, upload, clear], queue=False)
            with gr.Tab("批量检测", id="batch"):
                gr.HTML(heading("批量检测", "一次上传多张晶圆图，连续分析并汇总结果。每批最多 100 张。"))
                files = gr.File(label="上传多张 PNG 晶圆图", file_count="multiple", file_types=[".png"], type="filepath")
                counters = gr.HTML(batch_counters(0))
                with gr.Row():
                    batch_button = gr.Button("开始批量检测", variant="primary", interactive=service.ready)
                    batch_clear = gr.Button("清空列表")
                status = gr.Markdown("等待上传晶圆图。")
                gr.HTML('<div class="section-heading">类别分布</div><div class="section-caption">仅统计检测成功的图片；未接受或失败的图片会单独记录。</div>')
                counts = gr.HTML(class_counts())
                gr.HTML('<div class="section-heading">检测明细</div>')
                batch_table = gr.Dataframe(headers=["文件名", "预测类别", "置信度", "状态"], interactive=False, elem_id="batch-results")
                export = gr.File(label="下载 CSV 检测报告", interactive=False)
                files.change(ui.upload_batch, inputs=[files], outputs=[counters, batch_table, counts, export, status], queue=False)
                batch_clear.click(lambda: (None, batch_counters(0), [], class_counts(), None, "等待上传晶圆图。"),
                                  outputs=[files, counters, batch_table, counts, export, status], queue=False)

                def start_batch(paths):
                    return (gr.Button(interactive=False), gr.File(interactive=False), gr.Button(interactive=False),
                            "正在逐张检测，请稍候。进度条显示已处理、成功和失败数量。", None,
                            batch_counters(len(paths or [])), [], class_counts())

                def finish_batch():
                    return gr.Button(interactive=service.ready), gr.File(interactive=True), gr.Button(interactive=True)

                def run_batch(paths, progress=gr.Progress()):
                    return ui.batch(paths, progress=progress)

                event = batch_button.click(start_batch, inputs=[files],
                                           outputs=[batch_button, files, batch_clear, status, export, counters, batch_table, counts], queue=False)
                event = event.then(run_batch, inputs=[files], outputs=[counters, batch_table, counts, export, status],
                                   concurrency_id="wafer-model", concurrency_limit=1, show_progress="full")
                event.then(finish_batch, outputs=[batch_button, files, batch_clear], queue=False)
                gr.HTML('<div class="notice">置信度为模型输出概率，不能直接解释为实际正确率。批量分类与单张检测采用一致的决策规则。</div>')
            with gr.Tab("模型性能报告", id="analysis"):
                gr.HTML(heading("模型性能报告", "查看已完成实验的真实表现，区分独立测试结果与验证集指标。"))
                gr.HTML(report_cards(cfg))
                gr.Markdown(experiment_markdown(cfg))
                gr.Markdown("## 九类缺陷识别表现\n\n" + metrics_note)
                gr.Dataframe(value=table, headers=["缺陷类别", "Precision", "Recall", "F1", "测试样本数"], interactive=False)
                gr.Markdown("## 混淆矩阵\n\n" + cm_note)
                if cm_display:
                    gr.Image(value=cm_display, label="独立测试集混淆矩阵", interactive=False)
                gr.HTML('<div class="notice">实验指标来自已保存的真实记录，不代表当前上传图片的正确率。消融对照仅比较已提供的验证集表现。</div>')
            with gr.Tab("关于系统", id="about"):
                gr.HTML(heading("关于系统", "了解产品功能、使用方式与适用范围。"))
                gr.Markdown(about_markdown())
        nav_single.click(lambda: gr.Tabs(selected="single"), outputs=tabs, queue=False)
        nav_batch.click(lambda: gr.Tabs(selected="batch"), outputs=tabs, queue=False)
        gr.HTML('<div class="product-footer">晶圆缺陷智能检测与分析系统 · 智能识别，辅助判断</div>')
    return demo
