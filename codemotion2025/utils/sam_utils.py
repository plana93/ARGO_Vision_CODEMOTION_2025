# sam2_utils.py
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch


def show_mask(mask, ax, obj_id=None, random_color=False):
    """Render mask overlay on given axis."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    """Plot positive/negative points on axis."""
    coords = np.asarray(coords)
    labels = np.asarray(labels)
    pos = coords[labels == 1]
    neg = coords[labels == 0]
    ax.scatter(pos[:, 0], pos[:, 1], marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg[:, 0], neg[:, 1], marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    """Draw bounding box on axis."""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def _to_binary_u8(mask, threshold: float = 0.5) -> np.ndarray:
    """Convert mask to uint8 binary array (0/255)."""
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().squeeze().float().cpu().numpy()
    else:
        mask = np.asarray(mask)
    mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
    if mask.dtype == np.bool_:
        return (mask.astype(np.uint8) * 255)
    return ((mask > threshold).astype(np.uint8) * 255)

def save_binary_mask(mask, path, threshold: float = 0.5):
    """Save binary mask to PNG."""
    if mask.shape[0] == 1:
        mask = mask[0]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bin_u8 = _to_binary_u8(mask, threshold)
    Image.fromarray(bin_u8, mode="L").save(path)

def save_image(image: np.ndarray, path: Path | str):
    """
    Save a numpy image (binary or RGB) to disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure valid format
    img = np.array(image)
    if img.ndim == 2:
        mode = "L"
    elif img.ndim == 3 and img.shape[2] == 3:
        mode = "RGB"
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    Image.fromarray(img.astype(np.uint8), mode=mode).save(path)

def _compute_all_sizes(folder_dir, frame_names):
    """Return list of (width, height) for each image."""
    return [Image.open(Path(folder_dir) / f).size for f in frame_names]

def resize_all_to_largest(folder_dir, frame_names):
    """Resize all images to match the largest one."""
    sizes = _compute_all_sizes(folder_dir, frame_names)
    best_size = max(sizes, key=lambda s: s[0] * s[1])
    for f in frame_names:
        img_path = Path(folder_dir) / f
        Image.open(img_path).resize(best_size).save(img_path)

def show_frame(folder_dir, frame_name, frame_idx=0, show_axis=True):
    """Display an image frame."""
    img_path = Path(folder_dir) / frame_name
    plt.figure(figsize=(9, 6))
    plt.title(f"Frame {frame_idx}")
    plt.imshow(Image.open(img_path))
    if not show_axis:
        plt.axis("off")

def show_annotation_preview(img, points_list, obj_ids, labels_list=None, title="Manual point selection preview"):
    """
    Display annotated points on an image, colored per object.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)
    ax.set_title(title)

    cmap = plt.get_cmap("tab10")

    for obj_id, obj_points in zip(obj_ids, points_list):
        obj_points = np.array(obj_points)
        color = cmap(obj_id)

        ax.scatter(
            obj_points[:, 0],
            obj_points[:, 1],
            color=color,
            marker='*',
            s=200,
            edgecolor='white',
            linewidth=1.25,
            label=f"Obj {obj_id}"
        )

    ax.legend()
    plt.show()

    print("--- Annotation summary ---")
    for i, obj in enumerate(points_list):
        print(f"Obj {obj_ids[i]} -> {obj}")

