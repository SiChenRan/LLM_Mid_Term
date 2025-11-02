import torch, torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from .model import TransformerEncoder, LMHead
from .data import WikiText2Dataset
from .utils import set_seed
import os
import json
from datetime import datetime
from .utils import count_parameters, param_running_stats, grad_global_norm, build_warmup_cosine_scheduler


def train():
    print("[train] setting seed...")
    set_seed(42)

    # 设备选择与信息（支持通过环境变量 GPU_ID 指定卡号）
    if torch.cuda.is_available():
        gpu_id_env = os.environ.get("GPU_ID")
        if gpu_id_env is not None:
            try:
                gpu_index = int(gpu_id_env)
            except ValueError:
                print(f"[train] 非法 GPU_ID='{gpu_id_env}'，回退为 0")
                gpu_index = 0
            if 0 <= gpu_index < torch.cuda.device_count():
                torch.cuda.set_device(gpu_index)
                device = torch.device(f"cuda:{gpu_index}")
            else:
                print(f"[train] 指定 GPU_ID={gpu_index} 超出范围(0..{torch.cuda.device_count()-1})，回退为 0")
                torch.cuda.set_device(0)
                device = torch.device("cuda:0")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print(f"[train] device={device}")

    # 如果是 CUDA，先检查可用显存，显存过低则回退到 CPU
    if device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            free_b, total_b = torch.cuda.mem_get_info()
            free_gb = free_b / (1024 ** 3)
            total_gb = total_b / (1024 ** 3)
            print(f"[train] using GPU: {gpu_name} | free={free_gb:.2f}GB / total={total_gb:.2f}GB")
            if free_b < 1 * 1024 ** 3:  # 小于 1GB 则回退到 CPU
                print("[train] GPU 可用显存过低，回退到 CPU 训练。")
                device = torch.device("cpu")
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[train] 查询 GPU 显存失败: {e}")
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
    amp_enabled = device.type == "cuda"

    print("[train] loading dataset (train split)...")
    dataset = WikiText2Dataset("train", seq_len=128)
    print(f"[train] dataset loaded: vocab_size={dataset.vocab_size}, seq_len={dataset.seq_len}")
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # 结果与检查点目录
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("results", "checkpoints"), exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    vocab_size = dataset.vocab_size
    print("[train] building model...")
    model = TransformerEncoder(vocab_size).to(device)
    head = LMHead(128, vocab_size).to(device)
    print("[train] model built.")
    params = list(model.parameters()) + list(head.parameters())
    # 参数统计并保存
    try:
        param_stats_model = count_parameters(model)
        param_stats_head = count_parameters(head)
        print(f"[train] parameters | encoder: {param_stats_model}, head: {param_stats_head}")
        with open(os.path.join("results", f"params-{run_tag}.json"), "w", encoding="utf-8") as f:
            json.dump({"encoder": param_stats_model, "head": param_stats_head}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[train] 保存参数统计失败: {e}")

    print("[train] preparing optimizer and criterion...")
    lr = float(os.environ.get("LR", "3e-4"))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "0.01"))
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_id).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    # 学习率调度器（warmup + cosine）
    steps_per_epoch = len(loader)
    max_epochs = int(os.environ.get("MAX_EPOCHS", "5"))
    max_steps_per_epoch = int(os.environ.get("MAX_STEPS_PER_EPOCH", "0"))
    effective_steps_per_epoch = max_steps_per_epoch if max_steps_per_epoch > 0 else steps_per_epoch
    total_steps = max(1, effective_steps_per_epoch * max_epochs)
    warmup_env = os.environ.get("WARMUP_STEPS")
    warmup_steps = int(warmup_env) if warmup_env is not None else max(10, int(0.1 * total_steps))
    min_lr_ratio = float(os.environ.get("MIN_LR_RATIO", "0.1"))
    scheduler = build_warmup_cosine_scheduler(opt, warmup_steps, total_steps, min_lr_ratio=min_lr_ratio)
    print(f"[train] lr scheduler: warmup_steps={warmup_steps}, total_steps={total_steps}, min_lr_ratio={min_lr_ratio}")
    print("[train] start training...")

    # 是否从检查点恢复
    resume_path = os.environ.get("RESUME_PATH")
    start_epoch_num = 1  # 以 1 为起始的 epoch 计数
    max_epochs = int(os.environ.get("MAX_EPOCHS", "5"))
    resumed = False

    # 辅助：保存检查点
    def save_checkpoint(tag, epoch, step):
        ckpt = {
            "model": model.state_dict(),
            "head": head.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "vocab_size": vocab_size,
            "stoi": dataset.stoi,
            "itos": dataset.itos,
        }
        path = os.path.join("results", "checkpoints", f"ckpt-{tag}-e{epoch}-s{step}.pt")
        torch.save(ckpt, path)
        return path

    metrics = {"epoch_losses": [], "epoch_lrs": []}

    # 如果提供了检查点路径，则加载并设置续训起点
    if resume_path:
        if not os.path.isfile(resume_path):
            print(f"[train] 指定的检查点不存在: {resume_path}，将从头训练")
        else:
            print(f"[train] 从检查点恢复: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            # 兼容性检查
            ckpt_vocab_size = ckpt.get("vocab_size")
            if ckpt_vocab_size is not None and ckpt_vocab_size != vocab_size:
                print(f"[train] 警告：检查点 vocab_size={ckpt_vocab_size} 与当前数据集 {vocab_size} 不一致")
            # 恢复权重与优化器/AMP状态
            model.load_state_dict(ckpt["model"]) 
            head.load_state_dict(ckpt["head"]) 
            try:
                opt.load_state_dict(ckpt["optimizer"]) 
            except Exception as e:
                print(f"[train] 加载优化器状态失败，使用新优化器: {e}")
            try:
                scaler.load_state_dict(ckpt["scaler"]) 
            except Exception as e:
                print(f"[train] 加载 AMP scaler 失败，使用新 scaler: {e}")

            # 续训从下一轮开始（当前逻辑按 epoch 末保存）
            ckpt_epoch = int(ckpt.get("epoch", 0))
            start_epoch_num = ckpt_epoch + 1
            resumed = True

            # 为了保持同一 run 的文件命名，尽量沿用原 tag
            base_name = os.path.basename(resume_path)
            try:
                if base_name.startswith("ckpt-") and "-e" in base_name:
                    orig_tag = base_name[len("ckpt-"):base_name.index("-e")]
                    run_tag = orig_tag
            except Exception:
                pass

            print(f"[train] 已恢复到 epoch={ckpt_epoch}，将从 epoch {start_epoch_num} 开始，run_tag={run_tag}")

    # 若检查点已达到或超过最大轮次，则无需继续
    if start_epoch_num > max_epochs:
        print(f"[train] 检查点 epoch 已达到/超过 MAX_EPOCHS={max_epochs}，不再继续训练。")
        final_tag = f"final-{run_tag}"
        final_path = os.path.join("results", f"model-{final_tag}.pt")
        torch.save({
            "model": model.state_dict(),
            "head": head.state_dict(),
            "vocab_size": vocab_size,
            "stoi": dataset.stoi,
            "itos": dataset.itos,
        }, final_path)
        with open(os.path.join("results", f"vocab-{final_tag}.json"), "w", encoding="utf-8") as f:
            json.dump({"stoi": dataset.stoi, "itos": dataset.itos}, f, ensure_ascii=False, indent=2)
        with open(os.path.join("results", f"metrics-{final_tag}.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[train] final model saved to {final_path}")
        return

    for epoch_num in range(start_epoch_num, max_epochs + 1):
        print(f"[train] epoch {epoch_num} started...")
        epoch_loss_sum = 0.0
        epoch_batches = 0
        for i, (x, y) in enumerate(tqdm(loader, desc=f"Epoch {epoch_num}", unit="batch")):
            # 将批次迁移到设备
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = (x != dataset.pad_id).unsqueeze(1).unsqueeze(2)

            # 前向与损失（AMP）
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = head(model(x, mask))
                loss = criterion(out.view(-1, vocab_size), y.view(-1))

            opt.zero_grad(set_to_none=True)
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

            # 调度器步进（按步）
            try:
                scheduler.step()
            except Exception:
                pass

            if (i + 1) % 200 == 0:
                current_lr = opt.param_groups[0]["lr"]
                gnorm = grad_global_norm(model)
                tqdm.write(f"[train] epoch {epoch_num} step {i+1}, loss={loss.item():.4f}, lr={current_lr:.6f}, gnorm={gnorm:.4f}")
            # 累积损失
            epoch_loss_sum += loss.item()
            epoch_batches += 1
            # 不再按步保存检查点，改为每轮结束保存一次
            # 若限制了每轮最大步数，用于快速验证
            if max_steps_per_epoch > 0 and (i + 1) >= max_steps_per_epoch:
                break
        print(f"Epoch {epoch_num}, Loss={loss.item():.4f}")
        # 记录每轮平均损失
        avg_loss = epoch_loss_sum / max(1, epoch_batches)
        metrics["epoch_losses"].append(avg_loss)
        # 记录每轮末学习率与可选参数统计
        try:
            current_lr = opt.param_groups[0]["lr"]
            metrics["epoch_lrs"].append(current_lr)
        except Exception:
            metrics["epoch_lrs"].append(None)
        if os.environ.get("PRINT_PARAM_STATS", "0") == "1":
            pr = param_running_stats(model)
            print(f"[train] param stats (encoder) sample: {pr[:3]}")
        # 写入中间指标，便于可视化
        try:
            with open(os.path.join("results", f"metrics-{run_tag}.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[train] 写入中间指标失败: {e}")
        # 每轮结束保存检查点
        p = save_checkpoint(run_tag, epoch_num, epoch_batches)
        print(f"[train] epoch checkpoint saved: {p}")

    # 训练完成后保存最终模型与词表、指标
    final_tag = f"final-{run_tag}"
    final_path = os.path.join("results", f"model-{final_tag}.pt")
    torch.save({
        "model": model.state_dict(),
        "head": head.state_dict(),
        "vocab_size": vocab_size,
        "stoi": dataset.stoi,
        "itos": dataset.itos,
    }, final_path)
    with open(os.path.join("results", f"vocab-{final_tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"stoi": dataset.stoi, "itos": dataset.itos}, f, ensure_ascii=False, indent=2)
    # 最终 metrics 另存为一个固定名，便于脚本查找
    with open(os.path.join("results", f"metrics-{final_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join("results", "metrics-latest.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[train] final model saved to {final_path}")


if __name__ == "__main__":
    os.environ["HF_DATASETS_CACHE"] = "./data/hf_cache"
    train()