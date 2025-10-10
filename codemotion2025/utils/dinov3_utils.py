
import os
import math
import torch
import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt 
import torchvision.transforms as T
import timm
import inspect


def to_numpy(feature_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a feature tensor (1, C, H, W) or (C, H, W) or (H, W) to numpy.
    Returns (C, H, W) or (H, W) numpy array.
    """
    if not torch.is_tensor(feature_tensor):
        raise TypeError("Expected a torch.Tensor")
    x = feature_tensor.detach().cpu()
    if x.ndim == 4:
        # assume (B, C, H, W)
        if x.size(0) != 1:
            raise ValueError(f"Expected batch size 1, got {x.size(0)}")
        x = x.squeeze(0)
    return x.numpy()

def get_dinov3_transform(resize_size: int = 224):
    # README dinov3: LVD-1689M usa mean/std ImageNet (0.485,0.456,0.406)/(0.229,0.224,0.225)
    return T.Compose([
        T.Resize((resize_size, resize_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

def _is_square(n: int) -> bool:
    s = int(math.isqrt(n))
    return s * s == n

def _strip_register_tokens(tokens: torch.Tensor, num_register_tokens: int | None = None) -> torch.Tensor:
    """
    tokens: (B, N_no_cls, C) where CLS has already been removed.
    Removes trailing register tokens so that sequence length becomes a perfect square (H*W).
    """
    B, N, C = tokens.shape
    # 1) If we know how many register tokens exist, drop them directly.
    if num_register_tokens is not None and num_register_tokens > 0 and N - num_register_tokens > 0:
        cand = N - num_register_tokens
        if _is_square(cand):
            return tokens[:, :cand, :]
        # fall through to inference if the hint didn't yield a square (safety)

    # 2) Try common register counts used in practice.
    for r in (0, 4, 8, 16):
        if N - r > 0 and _is_square(N - r):
            return tokens[:, :N - r, :]

    # 3) Last-ditch: trim whatever is needed to reach the nearest lower square.
    side = int(math.isqrt(N))
    cand = side * side
    if cand <= 0:
        raise ValueError(f"Cannot make square from sequence length {N}.")
    return tokens[:, :cand, :]


def load_dinov3(
    model_variant: str = "vit_base_patch16_dinov3.lvd_1689m",
    resize_size: int = 224
):
    """
    Same docstring as before…
    """
    transform = get_dinov3_transform(resize_size)
    model = timm.create_model(model_variant, pretrained=True)
    model.eval()

    # Try to read number of register tokens if exposed by timm
    timm_num_regs = getattr(model, "num_register_tokens", None)

    if hasattr(model, "forward_features"):
        
        def _call_ff(x):
            sig = inspect.signature(model.forward_features)
            kwargs = {}
            if "return_all_tokens" in sig.parameters:
                kwargs["return_all_tokens"] = True
            if "return_dict" in sig.parameters:
                kwargs["return_dict"] = True
            out = model.forward_features(x, **kwargs)
            if isinstance(out, dict):
                tok = out.get("x", None) or out.get("tokens", None)
                if tok is None and "feature_maps" in out:
                    fm = out["feature_maps"][-1]  # (B,C,h,w)
                    return ("map", fm)
                if tok is None:
                    raise ValueError("forward_features dict non contiene 'x'/'tokens' né 'feature_maps'.")
                return ("tokens", tok)
            if isinstance(out, torch.Tensor) and out.ndim == 3:
                return ("tokens", out)
            if isinstance(out, torch.Tensor) and out.ndim == 4:
                return ("map", out)
            raise ValueError(f"Output inatteso da forward_features: type={type(out)}, shape={getattr(out,'shape',None)}")
    else:
        def _call_ff(x):
            y = model(x)
            if isinstance(y, torch.Tensor) and y.ndim == 4:
                return ("map", y)
            raise ValueError("Modello timm non espone forward_features e non produce feature map 4D.")

    def forward(t: torch.Tensor):
        with torch.no_grad():
            kind, out = _call_ff(t)
            if kind == "map":
                return out  # (B,C,h,w)
            # kind == "tokens": (B, N, C) includes CLS + possible registers
            # 1) drop CLS
            out = out[:, 1:, :]
            # 2) drop register tokens (trailing) so N becomes H*W
            out = _strip_register_tokens(out, num_register_tokens=timm_num_regs)
            return out  # (B, HW, C)

    def postproc(y: torch.Tensor):
        if y.ndim == 3:
            # (B, HW, C) tokens already stripped to square
            b, hw, c = y.shape
            if b != 1: raise ValueError(f"Expected B=1, got {b}")
            side = int(math.isqrt(hw))
            feat = y[0].transpose(0, 1).reshape(c, side, side)  # (C,H,W)
            return to_numpy(feat)
        elif y.ndim == 4:
            if y.size(0) != 1:
                raise ValueError(f"Expected B=1, got {y.size(0)}")
            return to_numpy(y)
        else:
            raise ValueError(f"Unexpected DINOv3 timm output shape: {y.shape}")

    return model, forward, transform, postproc
