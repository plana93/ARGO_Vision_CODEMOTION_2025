import os
import re
import hashlib
import mimetypes
from typing import Optional, Tuple, Dict

import requests
from requests.adapters import HTTPAdapter, Retry
from PIL import Image, ImageOps
from io import BytesIO

# ----------------------------------
# Environment utilities (Colab aware)
# ----------------------------------

def is_colab() -> bool:
    """Return True if running inside Google Colab."""
    try:
        import google.colab  # noqa
        return True
    except Exception:
        return False


# ----------------------------------
# Filesystem helpers
# ----------------------------------

def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def sanitize_filename(name: str, replacement: str = "_") -> str:
    """
    Sanitize a filename by removing/switching unsafe characters.
    Keeps the extension as-is.
    """
    name = name.strip()
    # Split base and ext while preserving leading dots in ext (if any)
    base, ext = os.path.splitext(name)
    base = re.sub(r'[^\w\-. ]+', replacement, base)  # keep alnum, _, -, ., space
    base = re.sub(r'\s+', replacement, base).strip(replacement)
    if not base:
        base = "downloaded_image"
    # Limit length a bit to avoid OS issues
    base = base[:120]
    if not ext:
        ext = ""
    return base + ext

def unique_path(path: str) -> str:
    """
    If 'path' exists, append _1, _2, ... before the extension to create a unique path.
    """
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    name,idx = base.split("_")
    #idx should 4digits
    i = int(idx) + 1
    while True:
        candidate = f"{name}_{str(i).zfill(4)}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


# ----------------------------------
# Image validation & processing
# ----------------------------------

def open_and_validate_image(data: bytes, convert_mode: Optional[str] = "RGB", verify: bool = True) -> Image.Image:
    """
    Open raw bytes as a PIL image, optionally verify, fix EXIF orientation, and convert mode.
    """
    bio = BytesIO(data)
    img = Image.open(bio)

    if verify:
        # Verify the file is not truncated/corrupted
        img.verify()
        # Reopen because verify() invalidates the parser
        bio.seek(0)
        img = Image.open(bio)

    # Fix EXIF orientation (common for JPEGs)
    img = ImageOps.exif_transpose(img)

    if convert_mode:
        # Only convert if a conversion is actually needed
        if img.mode != convert_mode:
            img = img.convert(convert_mode)

    return img


# ----------------------------------
# HTTP download helpers
# ----------------------------------

_DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (image-downloader/1.0; +https://example.com)"
}

def build_requests_session(
    retries: int = 2,
    backoff_factor: float = 0.3,
    status_force_list: Tuple[int, ...] = (429, 500, 502, 503, 504),
    headers: Optional[Dict[str, str]] = None,
) -> requests.Session:
    """
    Create a requests.Session with retry strategy and default headers.
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_force_list,
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(headers or _DEFAULT_HEADERS)
    return session

def normalize_url(url: str) -> str:
    """
    If the URL has no scheme, assume https://.
    """
    url = url.strip()
    if not re.match(r'^https?://', url, re.IGNORECASE):
        url = "https://" + url
    return url

def infer_extension_from_headers(content_type: Optional[str]) -> Optional[str]:
    """
    Infer an extension ('.jpg', '.png', ...) from a Content-Type header.
    """
    if not content_type:
        return None
    ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
    # Normalize some common mismatches
    if ext == ".jpe":
        ext = ".jpg"
    return ext

def is_image_content_type(content_type: Optional[str]) -> bool:
    """Return True if content type looks like an image."""
    return bool(content_type and content_type.lower().startswith("image/"))

def download_bytes_with_limits(
    url: str,
    session: Optional[requests.Session] = None,
    timeout: int = 15,
    max_mb: float = 20.0
) -> Tuple[bytes, Optional[str]]:
    """
    Download URL into memory (bytes) with a size cap and return (bytes, content_type).
    Streams data to avoid loading unbounded content in one go.
    """
    sess = session or build_requests_session()
    with sess.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        content_type = r.headers.get("Content-Type")
        # Optional: check content length early
        if "Content-Length" in r.headers:
            size = int(r.headers["Content-Length"])
            if size > max_mb * 1024 * 1024:
                raise ValueError(f"File too large: {size/1024/1024:.2f} MB (max {max_mb} MB)")
        # Stream to buffer with manual cap
        buf = BytesIO()
        downloaded = 0
        cap = int(max_mb * 1024 * 1024)
        for chunk in r.iter_content(chunk_size=8192):
            if not chunk:
                continue
            downloaded += len(chunk)
            if downloaded > cap:
                raise ValueError(f"Download exceeded {max_mb} MB cap")
            buf.write(chunk)
        return buf.getvalue(), content_type


# ----------------------------------
# Filenaming strategy
# ----------------------------------

def filename_from_url(url: str) -> str:
    """
    Extract a candidate filename from the URL path.
    """
    name = url.split("?")[0].split("#")[0].rstrip("/").split("/")[-1]
    return name or "downloaded_image"

def content_hash_name(data: bytes, ext: str = ".png") -> str:
    """
    Create a filename from the SHA1 hash of the content.
    """
    h = hashlib.sha1(data).hexdigest()[:12]
    return f"img_{h}{ext}"


# ----------------------------------
# Main entry points
# ----------------------------------

def _download_image_from_url(
    url: str,
    out_dir: str = "downloads",
    filename: Optional[str] = None,
    convert_mode: Optional[str] = "RGB",
    verify: bool = True,
    max_mb: float = 20.0,
    timeout: int = 15
) -> Tuple[str, Image.Image]:
    """
    Download an image from the web, validate it, and save to disk.

    Returns:
        (saved_path, PIL.Image.Image)
    """
    ensure_dir(out_dir)
    url = normalize_url(url)

    # Build session and fetch
    session = build_requests_session()
    data, content_type = download_bytes_with_limits(url, session=session, timeout=timeout, max_mb=max_mb)

    # Basic image content guard
    if not is_image_content_type(content_type):
        # Still try to open; some servers omit/lie on content-type
        pass

    # Open & validate image
    img = open_and_validate_image(data, convert_mode=convert_mode, verify=verify)

    idx = "0000"
    _ext = infer_extension_from_headers(content_type) or ".png"
    name = "img_" + idx + _ext
    

    # If name looks generic or still missing, fallback to content hash
    base, ext = os.path.splitext(name)
    if not ext:
        ext = ".png"
        name = base + ext

    # Ensure unique path
    save_path = unique_path(os.path.join(out_dir, name))

    # Save using PIL (re-encode to chosen format)
    try:
        img.save(save_path)
    except OSError:
        # Some formats (like webp on old PIL) may fail; fallback to jpg
        save_path = unique_path(os.path.join(out_dir, base + ".jpg"))
        img.save(save_path, format="jpg")

    return save_path, img


def download_image_from_url(
    prompt: str = "Enter the URL of an online image (or leave blank to use Colab upload): ",
    use_colab_upload_if_empty: bool = True,
    out_dir: str = "downloads"
) -> Optional[str]:
    """
    Interactive helper:
      - Ask for a URL; if provided, download it and return the saved path.
      - If empty and running on Colab (and allowed), open a file uploader and save locally.
      - Returns the path to the saved image or None if nothing was provided.
    """
    image_url = input(prompt).strip()

    if image_url:
        try:
            path, _ = _download_image_from_url(image_url, out_dir=out_dir)
            print(f"Downloaded and saved image as: {path}")
            return path
        except Exception as e:
            print(f"Error downloading image from URL: {e}")
            return None

    # If empty: optionally fall back to Colab upload
    if use_colab_upload_if_empty and is_colab():
        try:
            from google.colab import files  # type: ignore
            print("No URL provided. Please upload an image file from your computer…")
            uploaded = files.upload()
            if not uploaded:
                print("No file uploaded.")
                return None

            # Save the first uploaded file into out_dir
            ensure_dir(out_dir)
            fname = next(iter(uploaded.keys()))
            raw = uploaded[fname]
            # Validate with PIL & re-save (so path is consistent)
            img = open_and_validate_image(raw, convert_mode="RGB", verify=True)
            path = unique_path(os.path.join(out_dir, sanitize_filename(fname)))
            img.save(path)
            print(f"Uploaded and saved image as: {path}")
            return path
        except Exception as e:
            print(f"Error handling Colab upload: {e}")
            return None

    print("No URL provided and not running in Colab (or upload disabled).")
    return None


# ----------------------------------
# Example usage (uncomment to run)
# ----------------------------------
# if __name__ == "__main__":
#     # Option A: Direct call (programmatic)
#     # path, img = download_image_from_url(
#     #     "https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d",
#     #     out_dir="downloads",
#     #     convert_mode="RGB",
#     #     max_mb=25.0,
#     #     timeout=20
#     # )
#     # print("Saved to:", path)
#
#     # Option B: Interactive (URL first, otherwise Colab upload if available)

