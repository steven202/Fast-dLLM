"""
DELA: Diffusion Editing Language Agent
工具函数
"""

import argparse
import torch
import random
import numpy as np
import logging
from typing import Optional


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None
):
    """设置日志"""
    handlers = [logging.StreamHandler()]
    
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


def get_device(device: str = "auto") -> str:
    """获取设备"""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """计算可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(n: int) -> str:
    """格式化数字"""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    else:
        return str(n)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")
