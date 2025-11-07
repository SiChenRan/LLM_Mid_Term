import json, os
import matplotlib.pyplot as plt

def load_metrics(path_candidates):
    for p in path_candidates:
        if os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f), p
    raise FileNotFoundError(f"No metrics file found in: {path_candidates}")

def main():
    # 优先读取稳定路径，其次读取最近 run_tag 的 metrics
    candidates = [
        os.path.join('results', 'metrics-latest.json')
    ]
    # 允许用户传入备用文件路径（环境变量）
    alt = os.environ.get('METRICS_PATH')
    if alt:
        candidates.insert(0, alt)
    metrics, used = load_metrics(candidates)
    print(f"[plot] using metrics: {used}")

    # 根据使用的指标文件名自动选择输出目录：
    #  - 若提供 metrics-final-<tag>.json / metrics-<tag>.json，则保存到 results/plots/<tag>/
    #  - 否则保存到 results/plots/
    base = os.path.basename(used)
    name = os.path.splitext(base)[0]
    tag = None
    if name.startswith('metrics-final-'):
        tag = name[len('metrics-final-'):]
    elif name.startswith('metrics-') and name != 'metrics-latest':
        tag = name[len('metrics-'):]
    default_dir = os.path.join('results', 'plots')
    out_dir = os.environ.get('PLOT_DIR', os.path.join(default_dir, tag) if tag else default_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 解析数据集标签（基于训练 run_tag 的格式：<dataset>_<lang>_<YYYYMMDD>_<HHMMSS>）
    dataset_tag = None
    if tag:
        parts = tag.split('_')
        if len(parts) >= 3 and parts[-2].isdigit() and len(parts[-2]) == 8 and parts[-1].isdigit() and len(parts[-1]) == 6:
            dataset_tag = '_'.join(parts[:-2])  # e.g., IWSLT2017_en-de
        else:
            dataset_tag = tag

    losses = metrics.get('epoch_losses', [])
    lrs = metrics.get('epoch_lrs', [])
    cfg = metrics.get('config', {})

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(range(1, len(losses)+1), losses, marker='o')
    axs[0].set_ylabel('Avg Loss per Epoch')
    axs[0].grid(True, ls='--', alpha=0.4)

    axs[1].plot(range(1, len(lrs)+1), lrs, marker='o', color='orange')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('LR')
    axs[1].grid(True, ls='--', alpha=0.4)

    # 组装标题，包含数据集标签与消融参数摘要
    title = None
    if dataset_tag:
        parts = []
        parts.append(dataset_tag)
        if cfg:
            parts.append(f"d{cfg.get('d_model','?')} ff{cfg.get('d_ff','?')} h{cfg.get('num_heads','?')} enc{cfg.get('enc_layers','?')} dec{cfg.get('dec_layers','?')} cross{1 if cfg.get('use_cross_attn', True) else 0}")
        title = ' | '.join(parts)
    else:
        if cfg:
            title = f"d{cfg.get('d_model','?')} ff{cfg.get('d_ff','?')} h{cfg.get('num_heads','?')} enc{cfg.get('enc_layers','?')} dec{cfg.get('dec_layers','?')} cross{1 if cfg.get('use_cross_attn', True) else 0}"
    if title:
        axs[0].set_title(title)
    plt.tight_layout()
    out_name = 'training_curves.png' if not dataset_tag else f'training_curves-{dataset_tag}.png'
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=150)
    print(f"[plot] saved figure to {out_path}")

if __name__ == '__main__':
    main()