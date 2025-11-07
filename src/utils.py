import torch, random, numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    import math
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def count_parameters(model, verbose=False):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        for n, p in model.named_parameters():
            print(f"{n:50s} | {tuple(p.shape)} | {p.numel()/1e6:.3f}M | requires_grad={p.requires_grad}")
    return {"total": total, "trainable": trainable}

@torch.no_grad()
def param_running_stats(model):
    stats = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        stats.append((name, p.data.mean().item(), p.data.std().item()))
    return stats

def grad_global_norm(model):
    import math
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += (p.grad.detach().float().norm(2) ** 2).item()
    return math.sqrt(total)

def build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    import math
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        # cosine decay from 1.0 -> min_lr_ratio
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
def count_parameters(model, verbose=False):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        for n, p in model.named_parameters():
            print(f"{n:50s} | {tuple(p.shape)} | {p.numel()/1e6:.3f}M | requires_grad={p.requires_grad}")
    return {"total": total, "trainable": trainable}

@torch.no_grad()
def param_running_stats(model):
    # 返回简单统计用于日志
    stats = []
    for name, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        stats.append((name, p.data.mean().item(), p.data.std().item()))
    return stats

def grad_global_norm(model):
    import math
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += (p.grad.detach().float().norm(2) ** 2).item()
    return math.sqrt(total)
