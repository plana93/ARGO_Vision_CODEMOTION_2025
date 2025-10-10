
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

@torch.no_grad()
def last_layer_patch_features(
    model: torch.nn.Module,
    x: torch.Tensor,
    apply_norm: bool = True,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Return the last block feature map as patch-aligned tensor (B, C, Hp, Wp) from a timm ViT (incl. DINOv3).

    What it does:
      1) Runs the backbone up to the last token sequence (forward_features).
      2) Removes ALL prefix tokens (CLS + register/distilled/etc).
      3) Reshapes patch tokens from (B, Np, C) to (B, C, Hp, Wp).
      4) Optionally applies channel-wise LayerNorm (DINOv3-style).

    Args:
      model: timm ViT-like model with .forward_features and .patch_embed.
      x:     input tensor (B, 3, H, W) already preprocessed to the right size.
      apply_norm: apply LayerNorm over channels if model has `model.norm` as nn.LayerNorm.
      verbose: print helpful debugging info.

    Returns:
      fmap: torch.Tensor of shape (B, C, Hp, Wp)
    """
    # 1) Get final token sequence: shape should be (B, N_all, C)
    seq = model.forward_features(x)
    if seq.dim() != 3:
        raise RuntimeError(f"Expected (B, N, C) from forward_features, got {tuple(seq.shape)}")
    B, N_all, C = seq.shape

    # 2) Infer patch size and patch grid (Hp, Wp)
    ps = _infer_patch_size(model)
    H, W = x.shape[-2:]
    if H % ps != 0 or W % ps != 0:
        raise RuntimeError(f"Input size {H}x{W} must be divisible by patch size {ps}.")
    Hp, Wp = H // ps, W // ps
    num_patches = Hp * Wp

    # 3) Remove ALL prefix tokens (CLS + register + any others).
    #    Prefer model.num_prefix_tokens if available; otherwise compute from counts.
    n_prefix: Optional[int] = getattr(model, "num_prefix_tokens", None)
    if n_prefix is None:
        n_prefix = N_all - num_patches

    if n_prefix < 0:
        raise RuntimeError(
            f"Total tokens ({N_all}) < expected patches ({num_patches}). "
            f"Check input resolution or patch size."
        )
    if n_prefix > 0:
        seq = seq[:, n_prefix:, :]  # keep only patch tokens

    # Now seq is (B, num_patches, C) → reshape to (B, C, Hp, Wp)
    if seq.shape[1] != num_patches:
        raise RuntimeError(
            f"After prefix removal, got {seq.shape[1]} tokens, expected {num_patches}."
        )

    fmap = seq.view(B, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()  # (B, C, Hp, Wp)

    # 4) Optional: DINOv3-style LayerNorm over channels
    if apply_norm and hasattr(model, "norm") and isinstance(model.norm, torch.nn.LayerNorm):
        fmap = fmap.permute(0, 2, 3, 1).contiguous()  # (B, Hp, Wp, C)
        fmap = F.layer_norm(fmap, (C,))
        fmap = fmap.permute(0, 3, 1, 2).contiguous()  # (B, C, Hp, Wp)

    if verbose:
        print(f"[last_layer_patch_features] x: {tuple(x.shape)}")
        print(f"[last_layer_patch_features] patch_size: {ps} | grid: {Hp}x{Wp} ({num_patches} patches)")
        print(f"[last_layer_patch_features] tokens: total={N_all}, prefix={n_prefix}, patch={seq.shape[1]}")
        print(f"[last_layer_patch_features] fmap: {tuple(fmap.shape)}  (C={C})")

    return fmap


def _infer_patch_size(model: torch.nn.Module) -> int:
    """
    Infer ViT patch size from timm model.patch_embed.
    Works for both conv-proj (has .proj.kernel_size) and explicit .patch_size attributes.
    """
    if not hasattr(model, "patch_embed"):
        raise RuntimeError("Model lacks 'patch_embed'; cannot infer patch size.")
    pe = model.patch_embed

    # Common case: conv projection with kernel_size=(ps, ps)
    if hasattr(pe, "proj") and hasattr(pe.proj, "kernel_size"):
        ks = pe.proj.kernel_size
        return ks[0] if isinstance(ks, (tuple, list)) else int(ks)

    # Some models expose patch_size directly
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
