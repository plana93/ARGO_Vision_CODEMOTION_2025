
import os
import math
import torch
import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt 
import torchvision.transforms as T
import torch.nn.functional as F

import timm
import inspect
import kornia.morphology as km
import kornia.contrib as kc

from torchvision.transforms import InterpolationMode

from typing import Optional, Tuple, Union

import torch.nn.functional as F

# @torch.no_grad()
# def last_layer_patch_features(
#     model: torch.nn.Module,
#     x: torch.Tensor,
#     apply_norm: bool = True,
#     verbose: bool = False,
# ) -> torch.Tensor:
#     """
#     Estrarre la feature map finale allineata alle patch da un modello ViT (DINOv3 compatibile),
#     in modo coerente con get_intermediate_layers(..., reshape=True).

#     Output shape: (B, C, Hp, Wp)
#     """

#     # 1️⃣ Ottieni sequenza di token (B, N_all, C)
#     seq = model.forward_features(x)
#     if seq.dim() != 3:
#         raise RuntimeError(f"Expected (B, N, C), got {tuple(seq.shape)}")
#     B, N_all, C = seq.shape

#     # 2️⃣ Recupera patch size dal modello
#     ps = _infer_patch_size(model)
#     H, W = x.shape[-2:]
#     Hp, Wp = H // ps, W // ps
#     num_patches = Hp * Wp

#     # 3️⃣ Calcola quanti token extra (CLS + register) ci sono
#     n_prefix = getattr(model, "num_prefix_tokens", None)
#     if n_prefix is None:
#         n_prefix = getattr(model, "n_storage_tokens", 0) + 1  # DINOv3 usa 5 token extra (1 cls + 4 reg)

#     # Sanity check
#     if N_all < num_patches + n_prefix:
#         raise RuntimeError(
#             f"Token mismatch: total={N_all}, expected >= {num_patches + n_prefix}"
#         )

#     # 4️⃣ Rimuovi token extra (CLS + registers)
#     seq = seq[:, n_prefix:, :]  # (B, num_patches, C)

#     # 5️⃣ Reshape → coerente con get_intermediate_layers(reshape=True)
#     fmap = seq.reshape(B, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()

#     # 6️⃣ (Opzionale) LayerNorm canale-wise come DINOv3
#     if apply_norm and hasattr(model, "norm") and isinstance(model.norm, torch.nn.LayerNorm):
#         fmap = fmap.permute(0, 2, 3, 1)  # (B, Hp, Wp, C)
#         fmap = F.layer_norm(fmap, (C,))
#         fmap = fmap.permute(0, 3, 1, 2).contiguous()  # (B, C, Hp, Wp)

#     # 7️⃣ Debug info
#     if verbose:
#         print(f"[last_layer_patch_features]")
#         print(f"  Input: {tuple(x.shape)}")
#         print(f"  Patch size: {ps} | Grid: {Hp}x{Wp} ({num_patches} patches)")
#         print(f"  Tokens: total={N_all}, prefix={n_prefix}, patch={seq.shape[1]}")
#         print(f"  Output fmap: {tuple(fmap.shape)} (C={C})")

#     return fmap




@torch.no_grad()
def extract_from_layer_dinov3(
    img_in: Union[Image.Image, torch.Tensor],
    model: torch.nn.Module,
    tfm: T.Compose,
    device: torch.device | str,
    mask_hw: Optional[torch.Tensor] = None,   # (H,W) on original image, 0/1 or 0..255
    pad_to_patch_multiple: bool = True,
    verbose: bool = False,
):
    """
    Extract last-layer feature map for DINOv3 (timm) with consistent preprocessing.
    - Accepts PIL.Image (recommended) or CHW tensor -> converts to PIL for the transform.
    - Applies `tfm` (keeps aspect ratio).
    - (Optional) Pads AFTER transform to make H,W divisible by patch size.
    Returns:
        * (C, Hp, Wp) if mask_hw is None
        * (C,)        if mask_hw provided (masked average embedding)
    """
    # 0) Ensure PIL for the transform
    if isinstance(img_in, torch.Tensor):
        if img_in.ndim != 3:
            raise ValueError("Tensor image must be CHW.")
        img_pil = T.functional.to_pil_image(img_in.cpu())
    elif isinstance(img_in, Image.Image):
        img_pil = img_in
    else:
        raise TypeError(f"Unsupported img type: {type(img_in)}")

    # 1) timm preprocessing (aspect ratio preserved)
    x = tfm(img_pil)  # (3, Ht, Wt)
    Ht, Wt = x.shape[-2:]

    # 2) optional padding to multiples of patch
    ps = _infer_patch_size(model)
    if pad_to_patch_multiple:
        pad_h = (ps - (Ht % ps)) % ps
        pad_w = (ps - (Wt % ps)) % ps
        if pad_h or pad_w:
            x = F.pad(x.unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate").squeeze(0)
            Ht, Wt = x.shape[-2:]

    if verbose:
        print(f"[extract_from_layer_dinov3]")
        print(f"  transform output: (3,{Ht},{Wt});  patch={ps} -> grid=({Ht//ps},{Wt//ps})")

    # 3) forward to patch features
    xb = x.unsqueeze(0).to(device)           # (1,3,Ht,Wt)
    fmap = last_layer_patch_features_v3(model, xb, apply_norm=True, verbose=verbose).squeeze(0)  # (C,Hp,Wp)

    if mask_hw is None:
        if verbose:
            print(f"  fmap: {tuple(fmap.shape)}  (C,Hp,Wp)")
        return fmap

    # 4) masked average (resize mask to (Hp,Wp))
    Hp, Wp = fmap.shape[1:]
    m = mask_hw.float()
    if m.max() > 1:
        m = (m > 127).float()  # binarize 0/255 -> 0/1

    m_patch = F.interpolate(
        m.unsqueeze(0).unsqueeze(0), size=(Hp, Wp), mode="nearest"
    ).squeeze(0).squeeze(0)  # (Hp,Wp)

    w = (m_patch > 0).float()
    denom = w.sum().clamp_min(1.0)
    emb = (fmap.view(fmap.shape[0], -1) * w.flatten()).sum(dim=1) / denom  # (C,)

    if verbose:
        print(f"  masked emb: {tuple(emb.shape)}  (C,)  pos_frac={w.mean().item():.4f}")
    return emb

# ---------------------------
# Feature extraction (tokens → fmap)
# ---------------------------
@torch.no_grad()
def last_layer_patch_features_v3(
    model: torch.nn.Module,
    x_bchw: torch.Tensor,
    apply_norm: bool = True,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Return the last-layer patch-aligned feature map (B, C, Hp, Wp) for DINOv3 (timm).
    Steps:
      1) model.forward_features → tokens (B, N_all, C)
      2) drop ALL prefix tokens (CLS + registers)
      3) reshape to (B, C, Hp, Wp)
      4) optional channel-wise LayerNorm (DINOv3-style)
    """
    ps = _infer_patch_size(model)
    H, W = x_bchw.shape[-2:]
    assert H % ps == 0 and W % ps == 0, f"Input {H}x{W} must be divisible by patch size {ps}."

    seq = model.forward_features(x_bchw)  # (B, N_all, C)
    if seq.dim() != 3:
        raise RuntimeError(f"Expected (B,N,C), got {tuple(seq.shape)}")

    B, N_all, C = seq.shape
    Hp, Wp = H // ps, W // ps
    n_patches = Hp * Wp
    n_prefix = getattr(model, "num_prefix_tokens", N_all - n_patches)

    if n_prefix < 0 or (N_all - n_prefix) != n_patches:
        raise RuntimeError(f"Token mismatch: total={N_all}, prefix={n_prefix}, patches={n_patches}")

    seq = seq[:, n_prefix:, :]  # (B, n_patches, C)
    fmap = seq.view(B, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()  # (B, C, Hp, Wp)

    if apply_norm and hasattr(model, "norm") and isinstance(model.norm, torch.nn.LayerNorm):
        fmap = fmap.permute(0, 2, 3, 1).contiguous()  # (B, Hp, Wp, C)
        fmap = F.layer_norm(fmap, (C,))
        fmap = fmap.permute(0, 3, 1, 2).contiguous()  # (B, C, Hp, Wp)

    if verbose:
        print(f"[last_layer_patch_features_v3]")
        print(f"  input: {tuple(x_bchw.shape)}  patch={ps}  grid={Hp}x{Wp} (={n_patches})")
        print(f"  tokens: total={N_all}, prefix={n_prefix}, kept={seq.shape[1]}")
        print(f"  fmap: {tuple(fmap.shape)}  (B,C,Hp,Wp)")
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


def get_dinov3_transform_preserving_aspect_ratio(model: torch.nn.Module, force_size: Optional[int] = None) -> T.Compose:
    """
    Build a preprocessing transform that preserves aspect ratio:
      - Resize the *shorter* side to `force_size` (or model's default input height)
      - ToTensor + Normalize with ImageNet mean/std
    Padding to multiples of patch is handled *after* this transform in feature extraction.
    """
    cfg = timm.data.resolve_model_data_config(model)
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std  = cfg.get("std",  (0.229, 0.224, 0.225))

    default_h = cfg.get("input_size", (3, 224, 224))[1]
    short_side = int(force_size) if force_size is not None else int(default_h)

    tfm = T.Compose([
        # Passing an int → resize short side to `short_side` (keeps aspect ratio)
        T.Resize(short_side, interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    print(f"[dinov3_transform] short_side={short_side}  keep_aspect_ratio=True  mean/std={mean}/{std}")
    return tfm

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

def load_model_dinov3(model_name: str, device: torch.device | str) -> torch.nn.Module:
    """
    Load a DINOv3 ViT backbone from timm and set it to eval mode.
    Examples:
      - 'vit_small_plus_patch16_dinov3_qkvb.lvd1689m'  (C=384)
      - 'vit_base_patch16_dinov3.lvd_1689m'            (C=768)
      - 'vit_large_patch16_dinov3_qkvb.lvd_1689m'      (C=1024)
    """
    model = timm.create_model(model_name, pretrained=True)
    model.eval().to(device)
    print(f"[load_dinov3] Loaded '{model_name}' on {device}.")
    return model


def load_dinov3(
    model_variant: str = "vit_base_patch16_dinov3.lvd_1689m",
    resize_size: int = 224
):
    """
    Same docstring as before…
    """
    transform = get_dinov3_transform(resize_size)

    model = load_model_dinov3(
        model_name = model_variant,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ) 

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


# ---------------------------
# Similarity + mask extraction
# ---------------------------
def compute_similarity_map(
    fmap_chw: torch.Tensor,     # (C, Hp, Wp)
    embedding_c: torch.Tensor,  # (C,)
    normalize: bool = True,
    tag: str = "S(Q|S_emb)"
) -> torch.Tensor:
    """Cosine similarity per patch with a single embedding."""
    C, Hp, Wp = fmap_chw.shape
    f2d = fmap_chw.view(C, -1)
    f2d = f2d / (f2d.norm(p=2, dim=0, keepdim=True) + 1e-8)
    emb = embedding_c / (embedding_c.norm(p=2) + 1e-8)
    sim = (f2d * emb[:, None]).sum(dim=0).view(Hp, Wp)
    if normalize:
        smin, smax = sim.amin(), sim.amax()
        sim = (sim - smin) / (smax - smin + 1e-12) if (smax > smin) else torch.zeros_like(sim)
    print(f"[sim-map {tag}] shape={tuple(sim.shape)}  min={sim.min().item():.4f}  max={sim.max().item():.4f}  mean={sim.mean().item():.4f}  std={sim.std().item():.4f}")
    return sim


def _maybe_blur(sim_hw: torch.Tensor, blur: bool) -> torch.Tensor:
    """Optionally apply a small Gaussian blur to the similarity map."""
    if not blur:
        return sim_hw
    s = torch.nn.functional.gaussian_blur(
        sim_hw.unsqueeze(0).unsqueeze(0), kernel_size=(5, 5), sigma=1
    ).squeeze(0).squeeze(0)
    return s


def _morphological_closing(mask_hw_u8: torch.Tensor, closing_kernel_size: int, kernel_size: Optional[int]) -> torch.Tensor:
    """Apply morphological closing if requested.
    Morphological closing = dilation followed by erosion.
    It fills small black holes and connects nearby white regions in a binary mask,
    helping to smooth object boundaries without shrinking the overall area.
    """
    if closing_kernel_size <= 0:
        return mask_hw_u8
    ksz = kernel_size or closing_kernel_size
    kernel = torch.ones(ksz, ksz, device=mask_hw_u8.device, dtype=torch.float32)
    closed = km.closing(mask_hw_u8.float().unsqueeze(0).unsqueeze(0), kernel)
    return closed.squeeze(0).squeeze(0).to(torch.uint8)

def _keep_largest_cc(mask_hw_u8: torch.Tensor, num_ccl_iterations: int) -> torch.Tensor:
    """Keep only the largest connected component (foreground label != 0).
    Useful when multiple disjoint regions are detected but only the main object
    (largest connected area) should be kept. All smaller regions are removed.
    """
    if not mask_hw_u8.any():
        return mask_hw_u8
    labels = kc.connected_components(mask_hw_u8.float().unsqueeze(0).unsqueeze(0),
                                     num_iterations=num_ccl_iterations).squeeze(0).squeeze(0).long()
    areas = torch.bincount(labels.flatten())
    if areas.numel() <= 1:
        return mask_hw_u8
    areas[0] = 0  # ignore background
    largest = torch.argmax(areas)
    kept = (labels == largest).to(torch.uint8)
    print(f"[mask] kept largest CC (label={largest.item()}) pos_ratio={kept.float().mean().item():.4f}")
    return kept

def _area_filter(mask_hw_u8: torch.Tensor, min_area: int, num_ccl_iterations: int) -> torch.Tensor:
    """Remove all connected components smaller than a given minimum area.
    
    This helps eliminate small noisy detections or isolated blobs that
    are unlikely to be part of the main object.
    """
    if min_area <= 0:
        return mask_hw_u8
    labels = kc.connected_components(mask_hw_u8.float().unsqueeze(0).unsqueeze(0),
                                     num_iterations=num_ccl_iterations).squeeze(0).squeeze(0).long()
    areas = torch.bincount(labels.flatten())
    keep = (areas >= max(int(min_area), 1)).nonzero().squeeze(1)
    keep = keep[keep != 0]  # drop background
    if keep.numel() == 0:
        print(f"[mask] area filtering removed all. min_area={min_area}")
        return torch.zeros_like(mask_hw_u8)
    filtered = torch.isin(labels, keep).to(torch.uint8)
    print(f"[mask] after area filtering: pos_ratio={filtered.float().mean().item():.4f}")
    return filtered

def _compute_threshold(sim_hw: torch.Tensor, tval: float, threshold_type: str) -> float:
    """
    Compute the similarity threshold according to a chosen strategy.
    
    Supported strategies:
      - "static": fixed threshold = tval
      - "mean_std": mean(sim) + tval * std(sim)
      - "max": tval * max(sim)
    The result defines the cutoff above which pixels are considered foreground.
    """
    if threshold_type == "static":
        thr = float(tval)
    elif threshold_type == "mean_std":
        thr = sim_hw.mean().item() + tval * sim_hw.std().item()
    elif threshold_type == "max":
        thr = tval * sim_hw.max().item()
    else:
        raise ValueError(f"Unknown threshold_type: {threshold_type}")
    print(f"[mask] mode=threshold type={threshold_type} thr={thr:.4f}")
    return thr

def _mask_by_threshold(sim_hw: torch.Tensor, thr: float) -> torch.Tensor:
    """Binarize similarity map by threshold."""
    return (sim_hw >= thr).to(torch.uint8)

def _mask_by_topk(sim_hw: torch.Tensor, percent: float) -> torch.Tensor:
    """
    Select the top-k% most similar locations from a similarity map.
    
    Keeps only the highest similarity pixels, useful when object scale
    is unknown or the similarity distribution is skewed."""
    k = max(int(sim_hw.numel() * percent), 1)
    topk_vals, _ = torch.topk(sim_hw.flatten(), k)
    thr = topk_vals[-1]
    print(f"[mask] mode=topk% k={k} thr={thr.item():.4f}")
    return (sim_hw >= thr).to(torch.uint8)


def extract_mask(
    sim_hw: torch.Tensor,                      # (Hp, Wp)
    mode: str = "threshold",                   # 'threshold' | 'topk%'
    *,
    tval: float = 0.5,                         # threshold OR percent (0..1) for topk%
    blur: bool = False,
    threshold_type: str = "static",            # 'static' | 'mean_std' | 'max'
    closing_kernel_size: int = 0,
    kernel_size: Optional[int] = None,
    min_area: int = 0,
    num_ccl_iterations: int = 100,
    keep_largest_cc: bool = False,
) -> torch.Tensor:
    """
    Convert a similarity map (Hp, Wp) into a binary mask (Hp, Wp).
    Steps:
      1) optional blur
      2) threshold or top-k%
      3) optional morphological closing
      4) optional keep largest CC
      5) optional area filtering
    """
    assert sim_hw.ndim == 2, "sim_hw must be (Hp, Wp)."

    # (1) optional blur
    s = _maybe_blur(sim_hw, blur=blur)

    # (2) binarization
    if mode == "threshold":
        thr = _compute_threshold(s, tval=tval, threshold_type=threshold_type)
        mask = _mask_by_threshold(s, thr)
    elif mode == "topk%":
        mask = _mask_by_topk(s, percent=tval)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # (3) optional morphological closing
    mask = _morphological_closing(mask, closing_kernel_size, kernel_size)

    # debug: foreground ratio
    pos_ratio = mask.float().mean().item()
    print(f"[mask] shape={tuple(mask.shape)}  pos_ratio={pos_ratio:.4f}")

    # (4) optional keep largest CC
    if keep_largest_cc:
        mask = _keep_largest_cc(mask, num_ccl_iterations)

    # (5) optional area filtering
    if min_area > 0:
        mask = _area_filter(mask, min_area, num_ccl_iterations)

    return mask



def draw_patch_grid(ax, H_img, W_img, Hp, Wp, color='white', lw=0.5, alpha=0.4):
    """Draw a Hp×Wp patch grid on an image of size H_img×W_img."""
    step_y = H_img / Hp
    step_x = W_img / Wp
    # horizontal lines
    for r in range(1, Hp):
        y = r * step_y
        ax.plot([0, W_img], [y, y], color=color, lw=lw, alpha=alpha)
    # vertical lines
    for c in range(1, Wp):
        x = c * step_x
        ax.plot([x, x], [0, H_img], color=color, lw=lw, alpha=alpha)

def compute_bbox_from_mask(mask_uint8):
    """Return (ymin, xmin, ymax, xmax) of the 1s in a H×W uint8 mask. If empty returns None."""
    ys, xs = np.where(mask_uint8 > 0)
    if ys.size == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
