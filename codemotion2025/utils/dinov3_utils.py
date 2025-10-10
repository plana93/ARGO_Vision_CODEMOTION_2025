
import os
import math
import torch
import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt 
import torchvision.transforms as T
import timm
import inspect



import torch
import torch.nn.functional as F
from typing import Tuple, Optional


import torch
import torch.nn.functional as F
from typing import Optional, Tuple

@torch.no_grad()
def last_layer_patch_features(
    model: torch.nn.Module,
    x: torch.Tensor,
    apply_norm: bool = True,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Estrarre la feature map finale allineata alle patch da un modello ViT (DINOv3 compatibile),
    in modo coerente con get_intermediate_layers(..., reshape=True).

    Output shape: (B, C, Hp, Wp)
    """

    # 1️⃣ Ottieni sequenza di token (B, N_all, C)
    seq = model.forward_features(x)
    if seq.dim() != 3:
        raise RuntimeError(f"Expected (B, N, C), got {tuple(seq.shape)}")
    B, N_all, C = seq.shape

    # 2️⃣ Recupera patch size dal modello
    ps = _infer_patch_size(model)
    H, W = x.shape[-2:]
    Hp, Wp = H // ps, W // ps
    num_patches = Hp * Wp

    # 3️⃣ Calcola quanti token extra (CLS + register) ci sono
    n_prefix = getattr(model, "num_prefix_tokens", None)
    if n_prefix is None:
        n_prefix = getattr(model, "n_storage_tokens", 0) + 1  # DINOv3 usa 5 token extra (1 cls + 4 reg)

    # Sanity check
    if N_all < num_patches + n_prefix:
        raise RuntimeError(
            f"Token mismatch: total={N_all}, expected >= {num_patches + n_prefix}"
        )

    # 4️⃣ Rimuovi token extra (CLS + registers)
    seq = seq[:, n_prefix:, :]  # (B, num_patches, C)

    # 5️⃣ Reshape → coerente con get_intermediate_layers(reshape=True)
    fmap = seq.reshape(B, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()

    # 6️⃣ (Opzionale) LayerNorm canale-wise come DINOv3
    if apply_norm and hasattr(model, "norm") and isinstance(model.norm, torch.nn.LayerNorm):
        fmap = fmap.permute(0, 2, 3, 1)  # (B, Hp, Wp, C)
        fmap = F.layer_norm(fmap, (C,))
        fmap = fmap.permute(0, 3, 1, 2).contiguous()  # (B, C, Hp, Wp)

    # 7️⃣ Debug info
    if verbose:
        print(f"[last_layer_patch_features]")
        print(f"  Input: {tuple(x.shape)}")
        print(f"  Patch size: {ps} | Grid: {Hp}x{Wp} ({num_patches} patches)")
        print(f"  Tokens: total={N_all}, prefix={n_prefix}, patch={seq.shape[1]}")
        print(f"  Output fmap: {tuple(fmap.shape)} (C={C})")

    return fmap


def _infer_patch_size(model: torch.nn.Module) -> int:
    """Inferisci patch size da un modello timm."""
    if not hasattr(model, "patch_embed"):
        raise RuntimeError("Model lacks 'patch_embed' (non-ViT).")

    pe = model.patch_embed
    if hasattr(pe, "proj") and hasattr(pe.proj, "kernel_size"):
        ks = pe.proj.kernel_size
        return ks[0] if isinstance(ks, (tuple, list)) else int(ks)

    if hasattr(pe, "patch_size"):
        return pe.patch_size if isinstance(pe.patch_size, int) else pe.patch_size[0]

    raise RuntimeError("Unable to infer patch size from model.patch_embed.")


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
