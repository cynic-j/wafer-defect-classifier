# 晶圆缺陷智能检测与分析系统：比赛交接文档

更新日期：2026-11-24。本文严格区分原项目能力、本次系统改造、负责人提供的真实实验记录和待验证事项。

## 1. 开源基础及版权

- 原项目：<https://github.com/ALEX8642/wafer-defect-classifier>
- 当前项目远程仓库：<https://github.com/cynic-j/wafer-defect-classifier>
- 原许可：MIT License，Copyright (c) 2026 Alex8642。根目录 LICENSE 完整保留。
- 原作者已实现：WM-811K 数据加载、one-hot 编码、增强和数据划分，ResNet-18/50、CBAM、Focal Loss / 加权 CE、AMP、TTA×8、Grad-CAM / Grad-CAM++、温度校准、类别阈值、风险统计及原始 Gradio Demo。
- 上述网络结构和算法不能表述为本团队原创。本次未修改 model.py、train.py、data.py、evaluate.py、calibrate.py 或 explain.py。

## 2. 本次实际改造

- 修改 `src/wafer/demo.py`：保留模型、TTA 和 Grad-CAM++ 调用接口；默认读取 Baseline-01；从 checkpoint 恢复模型架构、CBAM 及输入尺寸等参数；检查类别映射。
- 加载完整权重时关闭 ImageNet 预训练下载，仅用现有 checkpoint。原始训练的 pretrained 参数单独保留用于展示。
- 温度与类别阈值文件仅从当前模型目录读取；无温度时 T=1、明确显示未校准；无阈值时 argmax。文件存在时检查数值合法性，但并不自动证明其与 checkpoint 的实验血缘或校准效果。
- 删除原 Demo 的原作者历史 macro-F1、ECE 展示。负责人提供的 Baseline-01 记录与上传结果独立展示。
- 新增 `src/wafer/demo_ui.py`：中文五页 Blocks + Tabs 界面、深蓝/白/浅灰工业风、功能导航、动态实验文件读取；产品页面不再展示运行环境版本。实验文件优先读取所选 outputs 目录，缺失时回退 `docs/wafer_evidence_20261124/<同名实验目录>/`；不修改资料副本。
- 单图：PNG 校验、真实分类、模型概率、Top-3、原图、Grad-CAM++ 热力图及叠加图、空间特征说明。
- 批量：最多 100 张，逐张真实分类，单图失败隔离，九类数量汇总、CSV 导出。批量不做 Grad-CAM 反向计算，分类复用相同模型、TTA、温度和阈值规则。
- 输入异常时拒绝推理，不输出貌似可靠的分类结论；none 不解释为晶圆合格，空间说明不解释为已验证根因。
- 同一模型推理加锁，避免并发 Grad-CAM hook 和梯度互相干扰。导出只写系统临时目录，历史实验目录保持只读。
- 启动不自动加载完整数据集或生成示例；只使用已有 `demo_examples/*.png` 缓存。无数据集时仍可在有权重的环境接受上传。`--preview` 显式进入静态预览，不加载权重且禁用检测按钮及后端检测回调，不能冒充推理成功。
- Demo 专用配置入口以 UTF-8 读取 YAML，避免 Windows 默认 GBK 解码失败，未改共享 config.py 和训练入口。
- 新增必要回归测试 `tests/test_competition_demo.py`，使用临时随机权重及合成输入；测试结果不是实验性能。

## 3. 数据与预处理

数据集：WM-811K。负责人/项目记录给出 811457 条记录，172950 条带标签记录。
服务器原始数据通常位于 `data/raw/LSWMD.pkl`；本次未读取服务器数据、未重新确认行数。
数据来源论文：Wu, Jang, Chen, *Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets*，IEEE TSM，2015。
精确下载 URL、数据 SHA256 和使用条款应在正式提交前补齐，不能从开源代码 MIT 许可推定数据集商业使用许可。

处理：筛选有标签记录 → 展开嵌套标签 → 固定九类映射 → 0/1/2 三通道 one-hot → 最近邻缩放 224×224。
训练使用随机 90° 旋转及水平/垂直翻转。优先采用 trainTestLabel 自带测试集，否则分层随机 70/10/20；验证集来自训练池。原代码未按 lot 分组，也未将划分索引持久保存。

界面 PNG 编码：outside=40、pass=160、fail=255。不直接接受原始 0/1/2 PNG、照片、SEM 图、配色图或带图例截图。
支持 8-bit 灰度、RGB 和不透明调色板 PNG；静态图最多 400 万像素。通道差异和灰度 LUT 检查用于拒绝明显不匹配输入，不能验证图像来源，也无法识别所有特意使用同配色的截图。

九类：Center、Donut、Edge-Loc、Edge-Ring、Loc、Near-full、Random、Scratch、none。
none 仅指未识别到已知缺陷空间模式，可能仍存在随机失效、未建模模式或分布外情况。

## 4. 服务器与开发环境

负责人提供的服务器信息：

| 项目 | 已知信息 |
|---|---|
| 系统 | Linux；发行版暂未提供 |
| 项目根目录 | `/home/darksky/jiang/wafer-defect-classifier` |
| GPU | NVIDIA GeForce RTX 3090 |
| PyTorch | 2.7.1+cu128 |
| CUDA | 12.8（负责人提供；驱动版本未提供） |
| Python / Gradio | 历史版本未提供，不猜测；启动后系统说明页读取当前实际版本 |

Codex 开发环境为独立 Windows 工作区，不与上述 Linux 共享文件系统。
本机 Python 3.12.7、PyTorch 2.11.0、torchvision 0.26.0、pytest 7.4.4，未检测到可用 CUDA，未安装 Gradio。
未安装或升级任何依赖。本机环境与训练环境不同，不作为服务器兼容性已通过的证据。

## 5. Baseline-01 训练与推理参数

| 参数 | 值 | 来源 |
|---|---|---|
| arch | resnet18 | 负责人确认 |
| cbam | true | 负责人确认 |
| loss | focal | 负责人确认 |
| batch_size | 32 | 负责人确认 |
| num_workers | 4 | 负责人确认 |
| seed | 42 | 负责人确认 |
| num_epochs | 40（上限，存在早停） | 负责人确认 |
| patience | 7 | 负责人确认 |
| input_size | 224 | 负责人确认 |
| lr | 0.001 | 负责人确认 |
| weight_decay | 0.0001 | 负责人确认 |
| focal_gamma | 2.0 | 本地 training_summary.txt 的损失日志 |
| cbam_reduction | 资料配置为 16，历史实际值以 checkpoint 为准 | 资料 baseline.yaml，未读取服务器 checkpoint |
| pretrained | false | 本地 training_summary.txt 启动日志 |
| 优化器 | 原代码 AdamW | 未修改的训练实现 |
| 调度器 | 原代码 CosineAnnealingLR | 同上 |
| CUDA AMP | 原代码启用 autocast 和 GradScaler | 同上 |
| 模型选择 | 验证集 macro-F1 最佳 | 同上 |
| TTA | true，D4 八视图，仅用于测试/推理 | 负责人确认 |
| 温度校准 | 未进行 | 负责人确认 |
| 类别阈值 | 未确认；界面显示实际文件加载状态，无文件使用 argmax | 需服务器核验 |

模型加载会恢复 checkpoint 中的必要配置；完整历史参数由维护人员读取 checkpoint 的 cfg 核对，产品页面不再展示参数表，不使用当前 baseline 冒充历史运行参数。
TTA 是推理选项；当前 Demo 按命令行/YAML 设置执行，不依据 checkpoint 的训练时 tta 字段擅自覆盖。

## 6. 真实实验记录与文件

以下数值由负责人提供，并已读取本地实验日志及 CSV 交叉核对。本次未重新训练、未重新评估，不是原作者 README 中的成绩。

| 实验 | 架构 | 最佳验证 Macro-F1 | 测试 Macro-F1 | Balanced Accuracy | 测试数量 |
|---|---|---:|---:|---:|---:|
| Baseline-01 | ResNet-18 + CBAM + Focal Loss | 0.9171 | 0.9139 | 0.8988 | 34590 |
| Ablation-01 | ResNet-18 + Focal Loss，无 CBAM | 0.9049 | 尚未评估 | 尚未评估 | 暂未提供 |

Baseline-01 独立测试为 TTA×8，未经温度校准。ECE、校准后概率可靠性、推理延迟和吞吐率均未提供，不展示虚构数值。

服务器实际产物（启动时动态读取，不改写）：

- `outputs/baseline_3090_gpu0_seed42/best.pt`
- `outputs/baseline_3090_gpu0_seed42/per_class_metrics.csv`
- `outputs/baseline_3090_gpu0_seed42/confusion_matrix.png`

九类 Precision/Recall/F1 优先从 outputs 下 CSV 读取，缺失时读取 docs 实验副本；不使用原作者 assets。CSV 列需为原 evaluate.py 输出：`class,precision,recall,f1_score,support`。
本地实际已读取以下资料：

- `docs/wafer_evidence_20261124/baseline_3090_gpu0_seed42/{training_summary.txt,evaluate.log,per_class_metrics.csv,confusion_matrix.png,baseline.yaml}`
- `docs/wafer_evidence_20261124/ablation_no_cbam_seed42/{training_summary.txt,ablation_no_cbam.yaml}`

CSV 原值如下（指标为资料中保留的四位小数）：

| 类别 | Precision | Recall | F1 | 样本数 |
|---|---:|---:|---:|---:|
| Center | 0.9756 | 0.9313 | 0.9529 | 859 |
| Donut | 0.8807 | 0.8649 | 0.8727 | 111 |
| Edge-Loc | 0.9037 | 0.8227 | 0.8613 | 1038 |
| Edge-Ring | 0.9775 | 0.9866 | 0.9820 | 1936 |
| Loc | 0.8988 | 0.7549 | 0.8206 | 718 |
| Near-full | 0.9667 | 0.9667 | 0.9667 | 30 |
| Random | 0.9143 | 0.9249 | 0.9195 | 173 |
| Scratch | 0.8739 | 0.8410 | 0.8571 | 239 |
| none | 0.9886 | 0.9966 | 0.9926 | 29486 |

评估日志确认最佳 checkpoint 为 epoch 27，验证 Macro-F1 为 0.9171；划分为 train=121065、val=17295、test=34590；T=1.0，未校准。已人工查看混淆矩阵原图。
资料中的 baseline.yaml 仍为 batch_size=128、output_dir=outputs，是配置副本而非最终命令行覆盖记录；负责人确认实际 batch_size=32，训练日志每轮 3784 批次也与 ceil(121065/32) 一致。未更改该副本；部署时完整参数以 best.pt 中 cfg 为准。

界面汇总分数从真实日志/CSV 解析，不在 Python 中内置实验成绩常量；文件均缺失或无法解析时显示“暂未提供”。
混淆矩阵只读复制到本次会话临时目录显示，不开放模型目录供下载。
更换为其他模型目录时不套用 Baseline-01 汇总指标；同名目录仍需要负责人保证权重和实验产物一致。主目录文件存在但损坏时明确报错，不以资料副本悄悄覆盖。

## 7. 消融对照及初步结论

验证集 macro-F1：0.9171 - 0.9049 = 0.0122，即 CBAM 组高 1.22 个百分点。
只能描述为这两次记录中的验证集差异，不能声称独立测试集提升、稳定统计显著性或产线收益。
Ablation-01 尚无独立测试结果；需核对其余训练条件相同并进行多随机种子对比后，才能更强地归因 CBAM 效果。
本次没有重新运行消融实验，没有修改 `configs/ablation_no_cbam.yaml`。

## 8. 技术架构与可用于 PPT 的改进点

```text
中文 Gradio Blocks / Tabs
  ├─ 系统首页：品牌入口、单张/批量导航、功能卡片与九类模式
  ├─ 单图：PNG 校验 → one-hot → 模型/TTA → argmax 或阈值 → Grad-CAM++
  ├─ 批量：逐张校验及分类 → 类别统计 → 临时 CSV
  ├─ 模型性能报告：当前目录真实 CSV / 混淆矩阵 + 验证集消融对照
  └─ 关于系统：产品背景、使用步骤、技术概述、许可与局限
```

可表述的系统改进：

1. 将原有模型能力整合为中文、面向晶圆状态图的五页可演示系统。
2. 分类结果、Top-3 和可解释热力图协同呈现，空间说明与根因诊断明确区分。
3. 严格输入契约和异常隔离，降低错误输入被包装为可靠结论的风险。
4. 增加批量筛查、数量汇总和可下载结果，便于人工复核。
5. 将实测记录、消融对照、当前运行状态分开标注，明确未校准概率的含义。

不能宣称为本团队新提出的 ResNet、CBAM、Focal Loss、TTA 或 Grad-CAM++ 算法。
不能宣称已有真实根因定位、在线产线闭环、自动判废、完整质量风险模型或经济收益验证。

## 9. 服务器启动与现场演示

在服务器已有、能够导入 torch/torchvision/gradio 的项目环境执行，不安装依赖，不启动训练：

```bash
cd /home/darksky/jiang/wafer-defect-classifier
# 先激活服务器实际使用的环境；环境名未知，不假定存在 .venv。
python -c "import sys,torch,torchvision,gradio; print(sys.version); print(torch.__version__,torchvision.__version__,gradio.__version__); print(torch.cuda.is_available())"
PYTHONPATH=src python -m wafer.demo --help
PYTHONPATH=src python -m wafer.demo --device cuda:0 --tta --port 7860 \
  --output-dir outputs/baseline_3090_gpu0_seed42
```

默认监听 127.0.0.1；从演示电脑运行：

```bash
ssh -L 7860:127.0.0.1:7860 darksky@<服务器实际地址>
```

浏览器打开 `http://127.0.0.1:7860`。地址和 SSH 用户以实际部署为准。
确需局域网访问可用 `--host 0.0.0.0`，是否对外开放由部署人员配置；不默认开启公开隧道。
启动默认模型目录也是 `outputs/baseline_3090_gpu0_seed42`，无需改 baseline.yaml。

Windows 本地仅预览（要求该环境已存在 Gradio；本次不安装依赖）：

```powershell
$env:PYTHONPATH = "src"
python -m wafer.demo --preview --device cpu --port 7860
```

此模式显示 docs 的真实实验资料，但明确标记未加载权重，不能检测。CLI 参数及配置加载兼容 Windows/Linux，不将 Linux 绝对路径当成本地可访问位置。

演示步骤：

1. 系统首页介绍产品用途，通过单张检测或批量检测进入操作。
2. 单图页上传一张原生 LUT PNG，先查看预览，再点击检测；等待分析后展示类别、模型置信度、Top-3 与三幅图，可点击清空重置。
3. 上传普通照片或错误编码图，展示拒绝检测；不要声称能识别所有分布外输入。
4. 批量页上传若干合法 PNG 和一张不合法 PNG，展示上传计数、实时进度、完成/失败计数、类别数量和 CSV 下载。
5. 模型性能报告页展示真实九类指标和混淆矩阵，说明无 CBAM 消融仅有验证成绩。
6. 关于系统页介绍技术概述、使用范围、开源来源和局限；参数、路径和环境信息仅留在维护文档。

现场前必须在服务器实际权重上完成合法单图、Grad-CAM++、批量导出和页面浏览验收。
没有温度文件时显示未校准是预期行为，不要为了消除标记临时生成或套用其他模型的 temperature.json。

## 10. 兼容性、验证与限制

- Gradio 要求 >=4；使用 File(filepath)、Blocks、Tabs、串行事件。Gradio 4/5 的 CSS 传给 Blocks，Gradio 6 的 CSS 传给 launch，由函数签名判断。
- 官方依据：<https://www.gradio.app/guides/gradio-6-migration-guide>；本机缺少 Gradio，不等于服务器启动已验证。
- 单图 `_run_inference` 返回 `(tensor, heatmap, pred_cls, probs)`，`_load_assets` 和 `build_demo` 返回结构保持兼容；原灰度辅助函数仍保留供已有测试使用。
- 同一模型串行处理，吞吐率未测定；批量无后台任务管理、取消与长期历史持久化。
- CSV 与图像加载于启动时读取，产物更新后需重启 Demo。
- 本次未修复训练/评估/校准模块中已有的校准与 TTA 口径、阈值联合最优化、按批次划分等研究问题，避免影响历史实验。
- 温度/阈值与 checkpoint 缺少自动哈希绑定，需人工保证对应关系；加载温度不展示虚构 ECE。
- 已知任务边界：单标签空间模式分类，非目标检测、分割、混合缺陷识别或真实工艺根因证明。
- 当前本地无服务器模型、真实输入、Gradio 和 CUDA，无法宣称本次已完成服务器真实模型推理、UI 启动或浏览器视觉验收。回归验证采用临时随机模型，不能当作 Baseline-01 性能结果。

## 11. 本机验证记录

最终原有 22 项测试及新增 16 项测试全部通过（38/38，33.55 秒），覆盖 UTF-8、静态预览拒绝推理、资料优先级、非法输入、类别统计、CSV 导出，以及临时随机 ResNet-18+CBAM 的 TTA×8 与 Grad-CAM++ 管线。合成输入和临时随机权重测试不代表 Baseline-01 真实推理或性能验证。
Python AST 语法检查通过，`python -m wafer.demo --help` 入口已执行成功。当前环境缺少 Gradio，无法完成实际网页启动及浏览器视觉验收；本机无真实 best.pt，不能声称 Baseline-01 真实推理测试通过。
本地 CSV、混淆矩阵及训练/评估摘要已真实读取，其读取验证不属于模型推理验证。
不安装依赖、不启动训练、不生成实验性能结果、不修改既有模型和历史产物。


## 12. 产品界面改造（2026-11-24）

本阶段以前端产品化为主，不改动已经验证的分类、TTA、Grad-CAM++、模型加载和输入校验算法。

- 页面改为系统首页、智能检测、批量检测、模型性能报告、关于系统。
- 深海军蓝品牌区、科技蓝操作按钮、青绿色强调色、浅灰白内容背景；卡片式布局，统一中文字体与间距。Logo 和晶圆结构图为代码绘制的品牌示意，并明确不是检测结果。
- 首页只保留产品介绍和检测入口，不展示实验分数、模型参数、服务器路径或运行版本。
- 智能检测采用上传/预览与结果左右分栏，增加清空按钮；上传校验和预览不运行模型。点击检测后显示加载状态并暂时禁用输入，避免结果与上传图片错配。原图、热力图、叠加图保留。
- 批量检测增加上传数量、进度追踪、已处理和失败数量、九类数量卡片、四列结果表和 CSV 下载。分类循环仅增加可选 progress 通知；原分类函数、数据和 CSV 列不变。开始新批次时清空旧结果，避免旧表被误认为本批结果。
- 模型性能报告仍动态读取真实文件，区分测试集与验证集，删除展示用绝对路径。缺失数据保持未提供，无 CBAM 独立测试不补数。
- 关于系统改为背景、功能、使用步骤、九类模式、技术概述、数据来源、局限与许可；开发参数不进入产品流程。
- Gradio 4/5 使用 show_api=False，Gradio 6 使用 footer_links=[]，并以 CSS 隐藏默认 Footer。不升级依赖。
- 错误提示保留可操作的格式说明，不向使用者输出文件系统路径或模型异常堆栈；CSV 仍保留完整逐图状态以供复核。

验收边界：本机仍未安装 Gradio、无 Baseline-01 权重及可用 CUDA。真实服务器推理和 Gradio 组件渲染需在已有服务器环境复验，不能将随机模型的回归测试当成 Baseline-01 实测。
已使用共享产品 HTML/CSS 创建临时静态排版页面，在浏览器检查 1920×1080、1366×900、1024×768 与 390×844 视口；静态布局未出现整页横向溢出，卡片按断点重排。这不等于完整 Gradio 页面与事件链已在浏览器验收。

产品化阶段最终验证：全部 44 项测试通过（49.70 秒），包括既有 38 项和新增 6 项产品回归。Python AST 语法检查通过，git diff --check 通过；对照 Git HEAD 的 AST，确认模型加载、单图推理、批量分类函数、PNG 解码与输入校验函数完全未变。批量服务只增加可选进度回调，CSV 导出格式保持不变。临时静态验收服务已停止，不改变服务器运行状态；原有未跟踪实验资料未被覆盖。
