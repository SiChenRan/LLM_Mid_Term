# LLMMidTerm — Seq2Seq Transformer (IWSLT2017 EN–DE)

本项目实现了一个简洁的 Seq2Seq Transformer，用于英德翻译（IWSLT2017 EN–DE）。代码包含训练、评估、保存结果与绘图脚本，支持通过环境变量控制关键超参数与消融开关，便于复现实验与扩展。

## Reproducibility and Code Structure

- GitHub Link: https://github.com/SiChenRan/LLM_Mid_Term
- Dependencies and environment setup
- Folder structure
- Command line examples
- Expected runtime and hardware used

### Dependencies and Environment Setup

- Recommended: Python 3.10
- Install dependencies:

```
conda create -n llm_mid_term python=3.10
conda activate llm_mid_term
pip install -r requirments.txt
```

Notes:
- 若需 GPU 加速，请根据本机 CUDA 版本选择合适的 PyTorch 安装方式（参考 https://pytorch.org/）。当前 `requirments.txt` 中默认 `torch==2.4.0`。
- 如无 GPU，训练会自动回退到 CPU，速度较慢。

### Folder Structure

```
LLMMidTerm/
├── .gitignore
├── README.md
├── config/
│   └── base.yaml                 # 目前训练通过环境变量配置，此文件供参考/扩展
├── data/
│   ├── IWSLT2017_EN_DE/          # 预处理后的 Arrow 文件（train/validation/test）
│   └── wikitext/wikitext-2-raw-v1/
├── requirments.txt               # 依赖列表（包含 torch/datasets/transformers 等）
├── results/                      # 训练与评估输出（metrics、params、plots、checkpoints、model）
│   ├── metrics-latest.json       # 每轮更新的稳定路径（绘图默认读取）
│   ├── metrics-final-<tag>.json  # 每次 run 的最终指标
│   ├── params-<tag>.json         # 参数统计
│   ├── vocab-final-<tag>.json    # 词表（stoi/itos）
│   ├── plots/                    # 训练曲线图输出位置
│   └── checkpoints/              # 按间隔保存的 checkpoint（已在 .gitignore 排除）
├── scripts/
│   └── run.sh                    # 统一入口：训练 + 绘图
└── src/
    ├── __init__.py
    ├── data.py                   # 数据集封装（IWSLT2017 / WikiText2）
    ├── layers.py                 # 注意力与 Transformer 层
    ├── model.py                  # Seq2SeqTransformer 模型实现
    ├── plot_metrics.py           # 读取 metrics 绘制训练曲线
    ├── train.py                  # 训练主程序（通过环境变量配置）
    └── utils.py                  # 工具函数（调度器/参数统计等）
```

### Command Line Examples

基本训练（默认英→德，5 epochs，写入 `results/`）：

```
python -u -m src.train
```

覆盖部分超参（通过环境变量）：

```
SRC_LANG=en TGT_LANG=de SEQ_LEN=128 BATCH_SIZE=32 \
D_MODEL=128 D_FF=512 NUM_HEADS=4 NUM_LAYERS=2 \
MAX_EPOCHS=5 LR=3e-4 LOG_INTERVAL=100 CKPT_INTERVAL=10 \
python -u -m src.train
```

使用统一脚本（训练后自动绘图，日志保存到 `results/train-<stamp>.log`）：

```
GPU_ID=0 MAX_EPOCHS=5 bash scripts/run.sh
```

仅绘图（从最终或最新指标文件绘制曲线）：

```
METRICS_PATH=results/metrics-final-<tag>.json python -u -m src.plot_metrics
# 或使用默认：python -u -m src.plot_metrics  # 自动读取 results/metrics-latest.json
```

输出文件说明（位于 `results/`）：
- `metrics-latest.json`：每轮更新，包含训练/验证损失与配置摘要（绘图默认读取）。
- `metrics-final-<run_tag>.json`：本次 run 的最终指标。
- `model-final-<run_tag>.pt`：最终模型权重（仅 state_dict）。
- `vocab-final-<run_tag>.json`：词表映射（stoi/itos）。
- `params-<run_tag>.json`：参数统计。
- `plots/`：训练曲线图（自动按 tag 分类保存）。
- `checkpoints/epoch-000X-<run_tag>.pt`：间隔保存的 checkpoint（默认每 10 个 epoch）。

### Configuration (Environment Variables)

训练通过环境变量配置，常用项及默认值：
- `SRC_LANG`=`en`，`TGT_LANG`=`de`
- `SEQ_LEN`=`128`，`BATCH_SIZE`=`32`，`NUM_WORKERS`=`2`
- `D_MODEL`=`128`，`D_FF`=`512`，`NUM_HEADS`=`4`
- `NUM_LAYERS`=`2`（若未显式设置 `ENC_LAYERS`/`DEC_LAYERS`，则两者均为该值）
- `ENC_LAYERS`（可选）、`DEC_LAYERS`（可选）
- `USE_CROSS_ATTN`=`True`（启用解码器交叉注意力）
- `MAX_EPOCHS`=`5`，`LR`=`3e-4`，`WEIGHT_DECAY`=`1e-4`
- `WARMUP_RATIO`=`0.1`，`GRAD_CLIP`=`1.0`
- `LIMIT_TRAIN_BATCHES`=`0`（视为未限制）、`LIMIT_VALID_BATCHES`=`0`
- `LOG_INTERVAL`=`100`（步级日志间隔），`CKPT_INTERVAL`=`10`
- `GPU_ID`（仅 `scripts/run.sh` 使用，用于设置 `CUDA_VISIBLE_DEVICES`）
- `PYTHON_BIN`（可选，指定 Python 解释器，默认为 `python`）

### Data Preparation

- IWSLT2017 EN–DE：代码期望在 `data/IWSLT2017_EN_DE/<split>/*.arrow` 下存在预处理后的 Arrow 文件（本仓库已包含示例结构）。
- WikiText-2：`src/data.py` 同时支持 `wikitext-2-raw-v1`，优先使用本地数据，其次尝试从 Hugging Face 加载（需联网）。本项目主要使用 IWSLT2017 进行翻译任务。

### Expected Runtime and Hardware

- 推荐使用单张 NVIDIA GPU（≥8–12 GB 显存）。在 `d_model=128, num_heads=4, num_layers=2, seq_len=128, batch_size=32, MAX_EPOCHS=5` 的默认配置下，单 GPU 约 **10–20 分钟** 完成；具体取决于 GPU 型号与数据 I/O。
- CPU 环境可运行但较慢，默认配置可能需要 **1–2 小时** 以上。
- 日志与曲线图可用于监控训练进度与学习率变化：`results/train-<stamp>.log`、`results/plots/*.png`。

---

Example (English):

```
Listing 2: Example training command
$ conda create -n transformer python=3.10
$ pip install torch matplotlib
$ python -m src.train
```

本项目使用环境变量进行配置，若需精确复现实验，请在运行命令前设置相应变量（见上文 Configuration 一节）。