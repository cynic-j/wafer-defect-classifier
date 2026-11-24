"""Chinese competition demo; keeps the original model/TTA/Grad-CAM++ pipeline.

Default: python -m wafer.demo --device cuda
Only reads trained assets. Runtime exports are written to a temporary directory.
PNG LUT: outside=40, pass=160, fail=255. No photos or SEM images.
"""
from __future__ import annotations

import json
import os
import tempfile
import csv
import inspect
import threading
import uuid
import dataclasses
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from wafer.config import WaferConfig, build_arg_parser, REPO_ROOT
from wafer.data import get_dataloaders, CLASS_NAMES
from wafer.model import build_model, CBAM
from wafer.explain import GradCAMPlusPlus, tensor_to_display
from wafer.evaluate import tta_predict, apply_thresholds

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "baseline_3090_gpu0_seed42"
_MODEL_LOCK = threading.Lock()  # Grad-CAM hooks/backward share a single model.


# ---------------------------------------------------------------------------
# Process-mode notes (shown alongside each prediction)
# ---------------------------------------------------------------------------

# Spatial observations only; these are not verified process root causes.
_PROCESS_NOTES = {
    "Center": "失效晶粒在晶圆中心附近聚集。",
    "Donut": "失效晶粒在中间半径处形成环带。",
    "Edge-Loc": "失效晶粒集中在边缘局部区域。",
    "Edge-Ring": "失效晶粒沿晶圆外围形成环状分布。",
    "Loc": "失效晶粒在局部非中心区域聚集。",
    "Near-full": "失效晶粒覆盖晶圆大部分有效区域。",
    "Random": "失效晶粒呈离散、无明显规则的分布。",
    "Scratch": "失效晶粒形成线状或弧状条带。",
    "none": "未识别到已知缺陷空间模式；不代表晶圆质量合格。",
}


# ---------------------------------------------------------------------------
# Model + temperature loading
# ---------------------------------------------------------------------------

def _load_assets(cfg: WaferConfig):
    """Load checkpoint, temperature scalar, and per-class thresholds.

    Returns (model, target_layer, temperature, thresholds).
    thresholds is an empty dict if thresholds.json does not exist yet.
    """
    ckpt_path = cfg.output_dir / "best.pt"
    temp_path = cfg.output_dir / "temperature.json"
    thresh_path = cfg.output_dir / "thresholds.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "请在服务器已有项目环境运行，或通过 --output-dir 指定已训练模型目录；不会自动训练。"
        )

    ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
    saved_cfg = ckpt.get("cfg", {})
    for key in ("arch", "cbam", "cbam_reduction", "input_size", "loss", "focal_gamma"):
        if key in saved_cfg:
            setattr(cfg, key, saved_cfg[key])
    if ckpt.get("class_to_idx") != {name: i for i, name in enumerate(CLASS_NAMES)}:
        raise ValueError("checkpoint 类别映射与系统九类顺序不一致，已停止加载。")
    # A complete checkpoint needs no ImageNet download, even if trained with it.
    training_pretrained = saved_cfg.get("pretrained", cfg.pretrained)
    cfg.pretrained = False
    model = build_model(cfg, num_classes=len(ckpt["class_to_idx"])).to(cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model._wafer_checkpoint_cfg = dict(saved_cfg)
    model._wafer_checkpoint_cfg["pretrained"] = training_pretrained
    model._wafer_checkpoint_info = {
        "epoch": ckpt.get("epoch"), "val_macro_f1": ckpt.get("val_macro_f1")
    }

    temperature = 1.0
    if temp_path.exists():
        with open(temp_path) as f:
            temperature = float(json.load(f)["temperature"])
        if not np.isfinite(temperature) or temperature < 0.05:
            raise ValueError("temperature.json 中温度必须为不小于 0.05 的有限数值。")
    else:
        print(f"Warning: {temp_path} not found — using T=1.0 (uncalibrated)")

    thresholds: dict = {}
    if thresh_path.exists():
        with open(thresh_path) as f:
            thresholds = json.load(f)
        if not isinstance(thresholds, dict) or any(
            name not in CLASS_NAMES or not isinstance(tau, (int, float))
            or not np.isfinite(tau) or not 0 <= tau <= 1
            for name, tau in thresholds.items()
        ):
            raise ValueError("thresholds.json 的类别或阈值无效。")
        print(f"  Per-class thresholds loaded ({len(thresholds)} classes)")

    target_layer = next(m for m in reversed(list(model.layer4.children()))
                        if not isinstance(m, CBAM))
    model._wafer_calibrated = temp_path.exists()
    return model, target_layer, temperature, thresholds


# ---------------------------------------------------------------------------
# Image decode: PNG → (3, H, W) one-hot tensor
# ---------------------------------------------------------------------------

def _lut_mismatch_fraction(img_array: np.ndarray, tol: int = 25) -> float:
    """
    Fraction of pixels whose gray value is not near the project LUT {40, 160, 255}.

    A rendered wafer-map PNG decodes to ~0; an arbitrary photo or screenshot has
    a continuous gray distribution and scores high. Used to warn on inputs the
    LUT decode in _png_to_tensor would silently turn into a garbage wafer map.
    """
    gray = img_array.mean(axis=2) if img_array.ndim == 3 else img_array.astype(float)
    near = np.zeros(gray.shape, dtype=bool)
    for v in (40, 160, 255):
        near |= np.abs(gray - v) <= tol
    return float(1.0 - near.mean())


def _png_to_tensor(img_array: np.ndarray, input_size: int = 224) -> torch.Tensor:
    """
    Reverse the project LUT (outside=40, pass=160, fail=255) back to {0,1,2},
    then encode as a (3, H, W) one-hot float tensor resized to input_size.
    """
    gray = img_array.mean(axis=2) if img_array.ndim == 3 else img_array.astype(float)
    # Midpoints: 40↔160 = 100, 160↔255 = 207.5 → 208
    wmap = np.where(gray < 100, 0, np.where(gray < 208, 1, 2)).astype(np.int64)
    t = torch.tensor(wmap, dtype=torch.long)
    tensor = F.one_hot(t, num_classes=3).permute(2, 0, 1).float()
    return F.interpolate(
        tensor.unsqueeze(0), size=(input_size, input_size), mode="nearest"
    ).squeeze(0)


# ---------------------------------------------------------------------------
# Inference + Grad-CAM
# ---------------------------------------------------------------------------

def _run_inference(
    img_array: np.ndarray,
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    temperature: float,
    thresholds: dict,
    cfg: WaferConfig,
):
    """
    Full pipeline: decode PNG → calibrated probs (TTA if enabled) + Grad-CAM++ heatmap.

    Two-pass design:
      Pass 1 — calibrated probabilities via TTA or single forward pass (no grad).
               Per-class thresholds applied to pick the predicted class.
      Pass 2 — Grad-CAM++ backward pass for the predicted class heatmap.
               Always single-pass (TTA gradients don't average meaningfully).

    Returns (tensor, heatmap, pred_cls, calibrated_probs).
    """
    _validate_image(img_array)
    tensor = _png_to_tensor(img_array, cfg.input_size)
    inp = tensor.unsqueeze(0).to(cfg.device)

    # Pass 1: calibrated probabilities
    if cfg.tta:
        cal_probs = tta_predict(model, tensor.unsqueeze(0), cfg.device, temperature)[0]
    else:
        with torch.no_grad():
            logits = model(inp)
        cal_probs = torch.softmax(logits / max(temperature, 0.05), dim=1).squeeze().cpu().numpy()

    # Apply per-class thresholds or plain argmax
    if thresholds:
        pred_cls = int(apply_thresholds(cal_probs[np.newaxis, :], thresholds, CLASS_NAMES)[0])
    else:
        pred_cls = int(cal_probs.argmax())

    # Pass 2: Grad-CAM++ for the predicted class
    with GradCAMPlusPlus(model, target_layer) as cam:
        heatmap, _, _ = cam.compute(inp, target_class=pred_cls)

    return tensor, heatmap, pred_cls, cal_probs


def _build_figure(
    tensor: torch.Tensor,
    heatmap: np.ndarray,
    pred_cls: int,
    cal_probs: np.ndarray,
    temperature: float,
    save_path: Path,
) -> str:
    """Render the 3-panel Grad-CAM figure, save to save_path, return the path string."""
    wafer_img = tensor_to_display(tensor)          # (H, W) uint8 {40, 160, 255}
    jet_map = plt.cm.jet(heatmap)[..., :3]         # (H, W, 3) float
    wafer_rgb = np.stack([wafer_img] * 3, axis=-1).astype(float) / 255.0
    blended = 0.55 * wafer_rgb + 0.45 * jet_map

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title, cmap in zip(
        axes,
        [wafer_img, heatmap, blended],
        ["Wafer map", "Grad-CAM", "Overlay"],
        ["gray", "jet", None],
    ):
        vmax = 255 if img.max() > 1.0 else 1.0
        ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    pred_name = CLASS_NAMES[pred_cls]
    conf = cal_probs[pred_cls]
    fig.suptitle(
        f"Predicted: {pred_name}  |  Model probability: {conf:.1%}  (T={temperature:.3f})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def _build_markdown(pred_cls, cal_probs, temperature, calibrated=False):
    from wafer.demo_ui import class_label
    top3 = cal_probs.argsort()[::-1][:3]
    rows = "\n".join(f"| {class_label(CLASS_NAMES[i])} | {cal_probs[i]:.2%} |" for i in top3)
    status = "已应用概率校准参数，可靠性仍需结合评估" if calibrated else "未校准，请结合人工复核"
    return (
        f"## {class_label(CLASS_NAMES[pred_cls])}\n\n"
        f"### 模型置信度：{cal_probs[pred_cls]:.2%}\n\n"
        f"{status}。置信度为模型输出概率，不等于实际正确率。\n\n"
        "### Top-3 候选类别\n\n| 类别 | 模型概率 |\n|---|---|\n" + rows +
        f"\n\n### 空间特征辅助分析\n\n{_PROCESS_NOTES[CLASS_NAMES[pred_cls]]}\n\n"
        "辅助分析仅描述空间分布，不是已验证的工艺根因诊断；预测不能用于自动放行。"
    )


# ---------------------------------------------------------------------------
# Example extraction (one wafer-map PNG per class, cached after first run)
# ---------------------------------------------------------------------------

def _extract_examples(cfg: WaferConfig, out_dir: Path) -> list[list]:
    """Return a list of [[image_path]] for each class, extracting from test set if needed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.png"))
    if len(existing) >= len(CLASS_NAMES):
        print(f"  Using cached examples from {out_dir}/")
        return [[str(p)] for p in existing]

    print("  Extracting demo examples from test set (one-time)...")
    _, _, test_loader, _, _ = get_dataloaders(cfg)

    collected: dict[int, torch.Tensor] = {}
    for inputs, labels in test_loader:
        for i in range(len(labels)):
            cls = int(labels[i].item())
            if cls not in collected:
                collected[cls] = inputs[i]
        if len(collected) == len(CLASS_NAMES):
            break

    from PIL import Image

    saved: list[list] = []
    for cls_idx in range(len(CLASS_NAMES)):
        if cls_idx not in collected:
            continue
        class_name = CLASS_NAMES[cls_idx]
        wafer_img = tensor_to_display(collected[cls_idx])  # (H, W) uint8 {40,160,255}
        save_path = out_dir / f"{class_name.lower().replace('-', '_')}.png"

        # Save as RGB PNG directly via PIL — no matplotlib rendering artifacts.
        # Values {40, 160, 255} are preserved exactly so _png_to_tensor decodes back
        # to the correct {0, 1, 2} map without border/title contamination.
        rgb = np.stack([wafer_img, wafer_img, wafer_img], axis=-1)
        Image.fromarray(rgb, "RGB").save(save_path)
        saved.append([str(save_path)])

    print(f"  Saved {len(saved)} examples to {out_dir}/")
    return saved


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def _validate_image(img_array):
    """Reject malformed or out-of-domain inputs before running the model."""
    if img_array is None:
        raise ValueError("请先上传晶圆图 PNG。")
    arr = np.asarray(img_array)
    if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[2] not in (1, 3)):
        raise ValueError("仅支持二维灰度或三通道 RGB 晶圆图。")
    if min(arr.shape[:2]) < 2 or arr.size > 12_000_000:
        raise ValueError("图片为空、尺寸过小或超过 1200 万通道像素的限制。")
    if not np.issubdtype(arr.dtype, np.number) or not np.isfinite(arr).all():
        raise ValueError("图片包含无效数值。")
    if arr.min() < 0 or arr.max() > 255:
        raise ValueError("请使用 8-bit 灰度编码（40、160、255）。")
    if arr.ndim == 3 and arr.shape[2] == 3:
        if np.max(np.ptp(arr.astype(float), axis=2)) > 3:
            raise ValueError("不支持彩色照片、SEM 配色图或截图；请使用规定灰度编码。")
    # Keep the original LUT diagnostic, with stricter acceptance for native PNGs.
    mismatch = _lut_mismatch_fraction(arr, tol=3)
    if mismatch > 0:
        raise ValueError(f"拒绝检测：{mismatch:.1%} 像素不符合灰度编码 40/160/255（容差 3）。")
    gray = arr.mean(axis=2) if arr.ndim == 3 else arr
    if not np.any(gray >= 157):
        raise ValueError("未找到有效晶粒区域，无法检测。")


def _read_png(path):
    from PIL import Image
    if not path:
        raise ValueError("请先上传晶圆图 PNG。")
    with Image.open(str(path)) as image:
        if image.format != "PNG":
            raise ValueError("仅支持 PNG 文件，请勿上传照片或其他格式。")
        if image.width * image.height > 4_000_000 or image.n_frames != 1:
            raise ValueError("只支持不超过 400 万像素的静态 PNG。")
        if image.mode not in ("L", "RGB", "P"):
            raise ValueError("仅支持 8-bit 灰度或不透明 RGB PNG。")
        if "transparency" in image.info:
            raise ValueError("请先将透明背景转换为灰度 40。")
        arr = np.asarray(image.convert("RGB"))
    _validate_image(arr)
    return arr


def _predict_probabilities(img_array, model, temperature, thresholds, cfg):
    """Batch classification without the unnecessary explanation backward pass."""
    _validate_image(img_array)
    tensor = _png_to_tensor(img_array, cfg.input_size)
    if cfg.tta:
        probs = tta_predict(model, tensor.unsqueeze(0), cfg.device, temperature)[0]
    else:
        with torch.no_grad():
            logits = model(tensor.unsqueeze(0).to(cfg.device))
        probs = torch.softmax(logits / temperature, dim=1)[0].cpu().numpy()
    pred = int(apply_thresholds(probs[None], thresholds, CLASS_NAMES)[0]) if thresholds else int(probs.argmax())
    return pred, probs


class DemoService:
    """Shared callbacks; lock also protects hooks across single and batch requests."""
    def __init__(self, cfg, model, layer, temperature, thresholds, tmp):
        self.cfg, self.model, self.layer = cfg, model, layer
        self.temperature, self.thresholds, self.tmp = temperature, thresholds, tmp
        self.calibrated = bool(getattr(model, "_wafer_calibrated", False))
        self.ready = model is not None

    def single(self, path):
        if not self.ready:
            return None, None, None, "静态预览未加载模型权重，不执行检测。"
        try:
            arr = _read_png(path)
            with _MODEL_LOCK:
                tensor, heatmap, pred, probs = _run_inference(
                    arr, self.model, self.layer, self.temperature, self.thresholds, self.cfg)
            if not np.isfinite(heatmap).all() or not np.isfinite(probs).all():
                raise ValueError("模型产生无效数值，请检查权重及运行环境。")
            wafer = tensor_to_display(tensor)
            heat_rgb = (plt.cm.jet(heatmap)[..., :3] * 255).astype(np.uint8)
            rgb = np.stack([wafer] * 3, axis=-1)
            overlay = (0.55 * rgb + 0.45 * heat_rgb).clip(0, 255).astype(np.uint8)
            md = _build_markdown(pred, probs, self.temperature, self.calibrated)
            return arr, heat_rgb, overlay, md
        except Exception as exc:
            return None, None, None, f"**检测未完成：** {str(exc)}\n\n未输出可靠检测结论。"

    def batch(self, paths, progress=None):
        from wafer.demo_ui import class_label
        counts = {name: 0 for name in CLASS_NAMES}
        rows = []
        if not self.ready:
            return [], [], None, "静态预览未加载模型权重，不执行检测。"
        if not paths:
            return [], [[class_label(k), 0] for k in counts], None, "请先上传 PNG 文件。"
        if len(paths) > 100:
            return [], [[class_label(k), 0] for k in counts], None, "单批最多 100 张，请分批检测。"
        if progress is not None:
            progress((0, len(paths)), desc=f"已处理 0/{len(paths)} 张 · 成功 0 · 失败 0")
        for index, path in enumerate(paths, 1):
            name = Path(str(path)).name
            try:
                arr = _read_png(path)
                with _MODEL_LOCK:
                    pred, probs = _predict_probabilities(arr, self.model, self.temperature, self.thresholds, self.cfg)
                if not np.isfinite(probs).all():
                    raise ValueError("模型输出无效数值")
                cls = CLASS_NAMES[pred]
                counts[cls] += 1
                rows.append([name, class_label(cls), float(probs[pred]), "成功", ""])
            except Exception as exc:
                rows.append([name, "", None, "拒绝/失败", str(exc)])
            if progress is not None:
                ok = sum(counts.values())
                progress((index, len(paths)), desc=f"已处理 {index}/{len(paths)} 张 · 成功 {ok} · 失败 {index-ok}")
        export = self.tmp / f"batch_{uuid.uuid4().hex}.csv"
        with export.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["文件名", "预测类别", "模型输出概率", "状态", "说明", "概率状态", "推理模式", "决策规则"])
            for row in rows:
                # Avoid spreadsheet formula execution from user-controlled filenames.
                safe = [("'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@", "\t", "\r")) else v) for v in row]
                writer.writerow(safe + ["温度参数已加载" if self.calibrated else "未校准", "TTA×8" if self.cfg.tta else "单次前向", "类别阈值" if self.thresholds else "argmax"])
        ok = sum(counts.values())
        return rows, [[class_label(k), v] for k, v in counts.items()], str(export), f"完成 {ok}/{len(paths)} 张；失败或拒绝 {len(paths)-ok} 张。计数仅统计成功结果，概率不等于质量保证。"


def build_demo(cfg: WaferConfig, preview=False):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("当前环境缺少 Gradio；请使用服务器已有的项目环境，本程序不会自动安装依赖。") from exc
    if int(gr.__version__.split(".")[0]) < 4:
        raise RuntimeError(f"需要 Gradio >=4，实际版本为 {gr.__version__}。")
    if preview:
        model, layer, temperature, thresholds = None, None, 1.0, {}
    else:
        model, layer, temperature, thresholds = _load_assets(cfg)
    tmp = Path(tempfile.mkdtemp(prefix="wafer_demo_"))
    os.environ["GRADIO_TEMP_DIR"] = str(tmp)
    from wafer.demo_ui import create_ui
    service = DemoService(cfg, model, layer, temperature, thresholds, tmp)
    demo = create_ui(gr, cfg, service)
    demo.queue(default_concurrency_limit=1)
    return demo, tmp


def _demo_config(args):
    """Read UTF-8 on Windows as well as Linux without changing training config."""
    import yaml
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("配置文件必须是 YAML 参数映射。")
    for field in dataclasses.fields(WaferConfig):
        value = getattr(args, field.name, None)
        if value is not None:
            raw[field.name] = value
    return WaferConfig(**raw)


def main():
    parser = build_arg_parser("晶圆缺陷智能检测与分析系统")
    parser.add_argument("--share", action="store_true", help="创建公开 Gradio 链接")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机")
    parser.add_argument("--preview", action="store_true", help="静态界面预览，不加载权重，禁用检测")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR
    try:
        cfg = _demo_config(args)
        app, demo_tmp = build_demo(cfg, preview=args.preview)
        from wafer.demo_ui import CSS
        kwargs = dict(server_name=args.host, server_port=args.port, share=args.share,
                      allowed_paths=[str(demo_tmp)])
        # Gradio 6 moved app-level CSS from Blocks to launch.
        if "css" in inspect.signature(app.launch).parameters:
            kwargs["css"] = CSS
        launch_params = inspect.signature(app.launch).parameters
        if "footer_links" in launch_params:
            kwargs["footer_links"] = []
        elif "show_api" in launch_params:
            kwargs["show_api"] = False
        app.launch(**kwargs)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"系统启动失败：{exc}\n")


if __name__ == "__main__":
    main()
