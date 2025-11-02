import torch, random, numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
