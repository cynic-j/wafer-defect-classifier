"""Competition UI and explicitly sourced experiment records; no model algorithms."""
from __future__ import annotations

import csv
import inspect
import platform
import re
import shutil
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
.gradio-container {max-width:1440px!important; background:#f3f6fa!important;
 font-family:'Microsoft YaHei','Noto Sans CJK SC',sans-serif!important; color:#152c46;
 --body-background-fill:#f3f6fa; --body-text-color:#152c46; --body-text-color-subdued:#50647b;
 --block-background-fill:#fff; --block-label-background-fill:#fff; --block-label-text-color:#15314e;
 --block-title-text-color:#15314e; --input-background-fill:#fff; --input-text-color:#152c46;
 --background-fill-primary:#fff; --background-fill-secondary:#edf2f7;
 --border-color-primary:#dce4ed; --border-color-accent:#1593ad;
 --button-secondary-background-fill:#edf2f7; --button-secondary-text-color:#15314e;}
#platform-header {background:#102d4e; color:white; padding:28px 32px;
 border-left:6px solid #24a8c3; border-radius:8px; margin-bottom:12px;}
#platform-header h1 {font-size:30px; margin:0 0 8px; color:white;}
#platform-header p {margin:5px 0; color:#d6e6f4; font-size:16px;}
.metric-grid {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:16px; margin:18px 0;}
.metric-card {background:white; border:1px solid #dce4ed; border-top:3px solid #1593ad;
 padding:20px; border-radius:6px; color:#15314e;}
.metric-card strong {display:block; font-size:30px; margin-top:8px;}
.metric-card small {color:#50647b;}
.gradio-container .primary {background:#155478!important; border-color:#155478!important; color:white!important;}
.gradio-container button {font-size:16px!important;}
.gradio-container .prose {font-size:16px; line-height:1.8;}
@media(max-width:800px) {.metric-grid {grid-template-columns:repeat(2,1fr);} #platform-header h1{font-size:24px;}}
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
        "### 已完成实验记录（动态读取真实实验资料）\n\n"
        "| 实验 | 模型 | 最佳验证 Macro-F1 | 独立测试 Macro-F1 | Balanced Accuracy |\n|---|---|---:|---:|---:|\n"
        f"| Baseline-01 | ResNet-18 + CBAM + Focal Loss | {metric(baseline['val'])} | {metric(baseline['test'])} | {metric(baseline['balanced'])} |\n"
        f"| Ablation-01 | ResNet-18 + Focal Loss，无 CBAM | {metric(ablation['val'])} | {metric(ablation['test'])} | {metric(ablation['balanced'])} |\n\n"
        f"Baseline-01 测试样本数：**{baseline['n'] if baseline['n'] is not None else '暂未提供'}**。"
        "负责人确认测试使用 TTA×8、尚未温度校准；当前实时推理状态见页首。\n\n"
        "上述是已保存的实验记录，不是当前上传图像的准确率，也不是本机重新评估结果。"
        "Ablation-01 独立测试尚未完成时，缺项保持未提供。\n\n"
        "来源：\n\n" + "\n".join(f"- `{p}`" for p in baseline['sources'] + ablation['sources'])
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
        return empty, "九类指标暂未提供；运行环境未找到当前模型目录中的 per_class_metrics.csv。"
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
        return result, f"来源：{path}（只读加载；优先 outputs，否则同名实验的 docs 资料副本）。"
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        return empty, f"九类指标暂未提供，CSV 读取失败：{exc}"


def runtime_markdown(gr):
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU（未检测到可用 CUDA）"
    return (f"运行环境实时读取：Python `{platform.python_version()}` · PyTorch `{torch.__version__}` · "
            f"Gradio `{gr.__version__}` · PyTorch CUDA `{torch.version.cuda}` · 设备 `{gpu}`。")


def system_description(cfg, service):
    saved = getattr(service.model, "_wafer_checkpoint_cfg", {})
    keys = ("arch", "cbam", "cbam_reduction", "loss", "focal_gamma", "batch_size", "num_workers",
            "seed", "num_epochs", "lr", "weight_decay", "patience", "input_size", "pretrained")
    rows = "\n".join(f"| {k} | {saved.get(k, 'checkpoint 未记录')} |" for k in keys)
    return """### 数据来源与处理

WM-811K：811457 条记录，其中 172950 条带标签记录。数据规模来源于项目数据记录，
当前上传文件不属于该统计。使用晶粒级状态图，非光学/SEM 图片。

处理流程：标签筛选 → 数据划分 → 0/1/2 三通道 one-hot → 最近邻缩放至模型输入尺寸 →
训练期旋转/翻转 → ResNet 分类 → 测试期可选 TTA×8 → 空间特征辅助分析。
优先采用原始 trainTestLabel 测试集，否则分层随机划分；验证集从训练池划分。

### 模型与技术来源

原开源项目已实现 ResNet、CBAM、Focal Loss、TTA、Grad-CAM++、温度校准和阈值工具。
本次完成中文五页系统、模型配置恢复、输入拒绝检查、批量统计及 CSV 导出、真实指标来源标注。
本次界面开发不训练模型，不执行校准，不修改历史实验文件。

### 已加载 checkpoint 的原始训练配置

| 参数 | checkpoint 记录 |
|---|---|
""" + rows + """

### 局限性

- 只支持九类单标签空间模式，不处理混合缺陷、目标框或像素级分割。
- 输入格式检查只能筛除明显不匹配的数据，不能证明图像来自真实产线；同配色截图仍可能通过。
- 未校准模型概率不能解释为实际正确率；温度参数文件存在也不代表 TTA 管线已验证校准。
- none 表示未识别到已知缺陷空间模式，不代表晶圆质量合格。
- 热力图为分类关注区域，不能当作缺陷精确边界或工艺根因证明。
- 未接入 MES、设备数据、批次溯源、自动放行或真实经济收益验证。
- 单批最多 100 张，顺序推理；临时 CSV 在服务会话期间可下载，未实现长期历史数据库。

### 开源许可

基于 [ALEX8642/wafer-defect-classifier](https://github.com/ALEX8642/wafer-defect-classifier)。
MIT License · Copyright (c) 2026 Alex8642；原 LICENSE 保留。
"""


def create_ui(gr, cfg, service):
    from wafer.demo import _PROCESS_NOTES
    kwargs = {"title": TITLE}
    if "css" in inspect.signature(gr.Blocks).parameters:
        kwargs["css"] = CSS
    table, metrics_note = load_class_metrics(cfg.output_dir)
    architecture = f"{cfg.arch.upper()} {'+ CBAM' if cfg.cbam else '（无 CBAM）'} + {cfg.loss.upper()}"
    calibration = f"温度参数已加载，T={service.temperature:.4f}；当前管线校准效果待验证" if service.calibrated else "未校准，T=1.0"
    decision = "已加载类别阈值" if service.thresholds else "无 thresholds.json，使用概率 argmax"
    mode = "TTA×8" if cfg.tta else "单次前向"
    # Copy display-only image into temporary storage; never serve the weights folder.
    cm = evidence_path(cfg.output_dir, "confusion_matrix.png")
    cm_display = None
    cm_note = "混淆矩阵暂未提供（当前模型目录未找到 confusion_matrix.png）。"
    if cm is not None:
        try:
            from PIL import Image
            with Image.open(cm) as image:
                image.verify()
            cm_display = str(service.tmp / "confusion_matrix.png")
            shutil.copyfile(cm, cm_display)
            cm_note = f"来源：{cm}；只读显示，优先 outputs，否则 docs 资料副本，未重新计算。"
        except (OSError, ValueError) as exc:
            cm_note = f"混淆矩阵暂未提供，图片读取失败：{exc}"
    with gr.Blocks(**kwargs) as demo:
        gr.HTML(f'<div id="platform-header"><h1>{TITLE}</h1><p>Wafer Defect Intelligence &amp; Analysis Platform</p><p>晶粒空间模式识别 · 可解释辅助分析 · 研究与比赛演示</p></div>')
        label = "已加载模型" if service.ready else "静态预览配置（未加载权重，检测已禁用）"
        gr.Markdown(f"**{label}：** {architecture}　｜　**推理设置：** {mode}　｜　{calibration}　｜　{decision}")
        with gr.Tabs() as tabs:
            with gr.Tab("系统总览", id="overview"):
                gr.Markdown("## 从晶圆空间模式到可解释的检测结果\n\n面向晶圆状态图的智能识别原型，为研究人员提供类别判断、候选概率和关注区域辅助分析。")
                if _record_context(cfg):
                    record = experiment_record(cfg.output_dir)
                    cards = [("测试集 Macro-F1", metric(record['test'])), ("Balanced Accuracy", metric(record['balanced'])),
                             ("独立测试样本", str(record['n']) if record['n'] is not None else "暂未提供"),
                             ("最佳验证 Macro-F1", metric(record['val']))]
                    gr.HTML('<div class="metric-grid">' + ''.join(
                        f'<div class="metric-card">{k}<strong>{v}</strong><small>Baseline-01 实验记录</small></div>'
                        for k, v in cards) + '</div>')
                gr.Markdown("上述模型级成绩从实验资料文件读取（优先 outputs，缺失时读取 docs 同名实验副本），与本次用户检测结果分开；负责人确认原测试为 TTA×8、未校准。\n\n"
                            "**WM-811K 数据集：** 811457 条晶圆记录、172950 条带标签记录；outside/pass/fail 晶粒级编码。\n\n"
                            "**支持类别：** " + "、".join(class_label(c) for c in CLASS_NAMES))
                with gr.Row():
                    nav_single = gr.Button("进入智能检测", variant="primary")
                    nav_batch = gr.Button("进入批量检测")
                    nav_analysis = gr.Button("查看模型分析")
                gr.Markdown("智能检测：单张 PNG 与热力图　｜　批量检测：逐张预测、类别汇总、CSV 下载　｜　模型分析：实验记录与消融对照　｜　系统说明：配置、流程与边界。")
            with gr.Tab("智能检测", id="single"):
                gr.Markdown("## 单张晶圆图检测\n\n仅上传原生灰度编码 PNG：**outside=40、pass=160、fail=255**。不接受普通照片、SEM 图、带标题/坐标轴截图。格式检查不等于来源真实性验证。")
                with gr.Row():
                    with gr.Column(scale=1):
                        upload = gr.File(label="上传晶圆图 PNG", file_types=[".png"], type="filepath")
                        button = gr.Button("开始检测", variant="primary", interactive=service.ready)
                        gr.Markdown("点击后才运行模型。拒绝或失败时清空上次图像和结果。")
                        examples = sorted((Path(cfg.output_dir) / "demo_examples").glob("*.png"))
                        if examples:
                            gr.Examples([[str(p)] for p in examples], inputs=[upload], label="已有示例缓存（不代表准确率评估）")
                    with gr.Column(scale=2):
                        result = gr.Markdown("请上传符合格式的 PNG，点击“开始检测”。")
                with gr.Row():
                    original = gr.Image(label="原始晶圆图", interactive=False)
                    heat = gr.Image(label="Grad-CAM++ 热力图", interactive=False)
                    overlay = gr.Image(label="关注区域叠加图", interactive=False)
                gr.Markdown("热力图在模型输入尺寸上生成；它解释最终类别的一次前向响应，不是 TTA 八视图热力图平均，也不是像素级检测结果。")
                button.click(service.single, inputs=[upload], outputs=[original, heat, overlay, result],
                             concurrency_id="wafer-model", concurrency_limit=1)
                upload.change(lambda: (None, None, None, "输入已变化，请点击“开始检测”。"),
                              outputs=[original, heat, overlay, result], queue=False)
            with gr.Tab("批量检测", id="batch"):
                gr.Markdown("## 批量晶圆图检测\n\n单批最多 100 张 PNG，逐张真实推理。批量省略 Grad-CAM++ 反向计算，分类规则与单张一致。无效图片单独记录，不计入类别数量。")
                files = gr.File(label="上传多张晶圆图", file_count="multiple", file_types=[".png"], type="filepath")
                batch_button = gr.Button("开始批量检测", variant="primary", interactive=service.ready)
                status = gr.Markdown("尚未运行。")
                batch_table = gr.Dataframe(headers=["文件名", "预测类别", "模型输出概率", "状态", "说明"], interactive=False)
                counts = gr.Dataframe(headers=["类别", "数量"], interactive=False)
                export = gr.File(label="下载本批 CSV（含失败记录）", interactive=False)
                batch_button.click(service.batch, inputs=[files], outputs=[batch_table, counts, export, status],
                                   concurrency_id="wafer-model", concurrency_limit=1)
                files.change(lambda: ([], [], None, "输入已变化，请重新运行批量检测。"),
                             outputs=[batch_table, counts, export, status], queue=False)
            with gr.Tab("模型分析", id="analysis"):
                model_status = "当前加载架构" if service.ready else "预览配置架构（未加载权重）"
                gr.Markdown(f"## 模型与实验分析\n\n{model_status}：**{architecture}**\n\n" + experiment_markdown(cfg))
                gr.Markdown("### 九类 Precision / Recall / F1\n\n" + metrics_note)
                gr.Dataframe(value=table, headers=["类别", "Precision", "Recall", "F1", "样本数"], interactive=False)
                gr.Markdown("### 混淆矩阵\n\n" + cm_note)
                gr.Image(value=cm_display, label="当前模型的真实混淆矩阵", interactive=False)
                gr.Dataframe(value=[[class_label(c), _PROCESS_NOTES[c]] for c in CLASS_NAMES],
                             headers=["类别", "空间特征说明（辅助分析）"], interactive=False)
            with gr.Tab("系统说明", id="about"):
                gr.Markdown(system_description(cfg, service))
                gr.Markdown(runtime_markdown(gr))
                gr.Markdown("Baseline-01 历史训练环境（负责人提供）：Linux / RTX 3090 / PyTorch 2.7.1+cu128 / CUDA 12.8。历史 Python、Gradio 版本未提供；上方仅为当前运行环境。")
        nav_single.click(lambda: gr.Tabs(selected="single"), outputs=tabs, queue=False)
        nav_batch.click(lambda: gr.Tabs(selected="batch"), outputs=tabs, queue=False)
        nav_analysis.click(lambda: gr.Tabs(selected="analysis"), outputs=tabs, queue=False)
        gr.Markdown("研究演示系统 · 模型概率不等于实际正确率 · 辅助分析不能替代人工复核和产线质量判定。")
    return demo
