"""
Image Matrix Utilities - Production-Ready Single-File Library
================================================================
Pure Python + NumPy + Pillow

Layers
------
1. Pure Python: no external dependencies (slow but portable).
2. NumPy: same functionality, vectorized and much faster.
3. Pillow (optional): load/save real images (PNG, JPEG, BMP, TIFF, WEBP -
   format is inferred automatically from the file extension by Pillow).

Backward compatibility: all function names/signatures from the previous
version are preserved. New functionality is additive.

Not included (by design - see the accompanying chat message for the
rationale): morphology (dilate/erode/opening/closing), noise generators,
drawing primitives, a pytest/ directory, and pyproject.toml packaging.
"""

from __future__ import annotations

import logging
import time
import tracemalloc
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

__version__ = "2.0.0"

logger = logging.getLogger("image_matrix_utils")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# ==================== Constants ====================

DEFAULT_THRESHOLD: float = 0.5
DEFAULT_DTYPE: str = "float32"
DEFAULT_CHANNELS: int = 3

SUPPORTED_DTYPES: dict[str, np.dtype] = {
    "uint8": np.dtype(np.uint8),
    "uint16": np.dtype(np.uint16),
    "int32": np.dtype(np.int32),
    "float16": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
    "bool": np.dtype(np.bool_),
}

# Standard fixed kernels (odd-sized, 2D)
SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
EMBOSS_KERNEL = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
LAPLACIAN_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
SOBEL_X_KERNEL = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_Y_KERNEL = SOBEL_X_KERNEL.T.copy()
PREWITT_X_KERNEL = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
PREWITT_Y_KERNEL = PREWITT_X_KERNEL.T.copy()
ROBERTS_X_KERNEL = np.array([[1, 0], [0, -1]], dtype=np.float32)
ROBERTS_Y_KERNEL = np.array([[0, 1], [-1, 0]], dtype=np.float32)


# ==================== Type Aliases ====================

PathLike = Union[str, Path]
ImageArray = np.ndarray  # shape (H, W), (H, W, 1), (H, W, 3) or (H, W, 4)
PurePythonMatrix = List[List[List[float]]]
DTypeName = Literal["uint8", "uint16", "int32", "float16", "float32", "float64", "bool"]
PaddingMode = Literal["reflect", "constant", "edge", "symmetric"]


# ==================== Exceptions ====================

class ImageMatrixError(ValueError):
    """Base exception for this library. Subclasses ValueError to remain
    backward-compatible with code that catches ValueError."""


class InvalidShapeError(ImageMatrixError):
    """Raised when dimensions (height/width/channels) are invalid."""


class InvalidDtypeError(ImageMatrixError):
    """Raised when an unsupported dtype is requested."""


class InvalidImageError(ImageMatrixError):
    """Raised when an image/matrix payload is empty, malformed, or a file
    path cannot be resolved."""


# ==================== Validation ====================

def _validate_dimensions(height: int, width: int, channels: int = 1) -> None:
    if height <= 0 or width <= 0 or channels <= 0:
        raise InvalidShapeError(
            f"Dimensions must be greater than zero (height={height}, "
            f"width={width}, channels={channels})"
        )


def _validate_dtype(dtype: str) -> np.dtype:
    if dtype not in SUPPORTED_DTYPES:
        raise InvalidDtypeError(
            f"Unsupported dtype: {dtype!r}. Supported: {sorted(SUPPORTED_DTYPES)}"
        )
    return SUPPORTED_DTYPES[dtype]


def _validate_numpy(matrix: np.ndarray) -> None:
    if not isinstance(matrix, np.ndarray):
        raise InvalidImageError(f"Expected np.ndarray, got {type(matrix).__name__}")
    if matrix.size == 0:
        raise InvalidImageError("Input matrix is empty")


def _validate_python(matrix: PurePythonMatrix) -> None:
    if not matrix or not matrix[0] or not matrix[0][0]:
        raise InvalidImageError("Input matrix is empty")


def _validate_path(path: PathLike) -> Path:
    return Path(path)


def _ensure_channels(matrix: ImageArray) -> ImageArray:
    """Normalize (H, W) / (H, W, 1) / (H, W, 3) / (H, W, 4) to (H, W, C)."""
    _validate_numpy(matrix)
    if matrix.ndim == 2:
        return matrix[:, :, None]
    if matrix.ndim == 3 and matrix.shape[-1] in (1, 3, 4):
        return matrix
    raise InvalidShapeError(
        f"Expected shape (H,W), (H,W,1), (H,W,3) or (H,W,4); got {matrix.shape}"
    )


def _validate_kernel(kernel: np.ndarray) -> None:
    if kernel.ndim != 2 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise InvalidShapeError(
            f"Kernel must be a 2D array with odd dimensions, got shape {kernel.shape}"
        )


# ==================== Pure Python API ====================

def create_image_matrix(
    height: int, width: int, channels: int = DEFAULT_CHANNELS, dtype: str = DEFAULT_DTYPE
) -> PurePythonMatrix:
    """
    Create an empty image matrix (all zeros) using pure Python.

    Parameters
    ----------
    height, width, channels : int
    dtype : {"float32", "uint8"}, default "float32"
        Pure-Python only supports these two; use the NumPy API for full
        dtype support.

    Returns
    -------
    list[list[list[float]]]

    Raises
    ------
    InvalidShapeError, InvalidDtypeError
    """
    _validate_dimensions(height, width, channels)
    if dtype not in ("float32", "uint8"):
        raise InvalidDtypeError(
            f"Pure Python API only supports 'float32'/'uint8', got {dtype!r}. "
            "Use create_image_matrix_np for other dtypes."
        )
    zero = 0.0 if dtype == "float32" else 0
    return [[[zero] * channels for _ in range(width)] for _ in range(height)]


def apply_threshold_filter(
    matrix: PurePythonMatrix, threshold: float = DEFAULT_THRESHOLD
) -> PurePythonMatrix:
    """
    Threshold filter (value > threshold => 1.0, else 0.0) using pure Python.

    Raises
    ------
    InvalidImageError
        If the matrix is empty.
    """
    _validate_python(matrix)
    height, width = len(matrix), len(matrix[0])
    channels = len(matrix[0][0])
    return [
        [
            [1.0 if matrix[i][j][c] > threshold else 0.0 for c in range(channels)]
            for j in range(width)
        ]
        for i in range(height)
    ]


# ==================== NumPy API ====================

def create_image_matrix_np(
    height: int, width: int, channels: int = DEFAULT_CHANNELS, dtype: DTypeName = DEFAULT_DTYPE
) -> ImageArray:
    """
    Create an empty image matrix using NumPy.

    Raises
    ------
    InvalidShapeError, InvalidDtypeError
    """
    _validate_dimensions(height, width, channels)
    np_dtype = _validate_dtype(dtype)
    return np.zeros((height, width, channels), dtype=np_dtype)


def apply_threshold_filter_np(matrix: ImageArray, threshold: float = DEFAULT_THRESHOLD) -> ImageArray:
    """
    Vectorized threshold filter. Same semantics as apply_threshold_filter.

    Raises
    ------
    InvalidImageError
        If the matrix is empty.
    """
    _validate_numpy(matrix)
    return (matrix > threshold).astype(np.float32)


# ==================== Pillow API ====================

def load_image_to_matrix(
    image_path: PathLike,
    target_size: Optional[Tuple[int, int]] = None,
    mode: Literal["RGB", "RGBA", "L"] = "RGB",
) -> ImageArray:
    """
    Load an image from disk into a normalized numpy matrix (0.0 - 1.0).
    Format is inferred automatically by Pillow from the file's contents.

    Parameters
    ----------
    target_size : tuple[int, int], optional
        (width, height) - matches PIL.Image.resize's own order.

    Raises
    ------
    ImportError, InvalidImageError
    """
    if not PIL_AVAILABLE:
        raise ImportError("Install pillow: pip install pillow")
    path = _validate_path(image_path)
    if not path.exists():
        raise InvalidImageError(f"Image file not found: {path}")
    img = Image.open(path)
    if target_size:
        img = img.resize(target_size)
    arr = np.array(img.convert(mode), dtype=np.float32) / 255.0
    logger.debug("Loaded image %s with shape %s", path, arr.shape)
    return arr


def save_matrix_to_image(matrix: Union[ImageArray, PurePythonMatrix], output_path: PathLike) -> None:
    """
    Save a matrix as an image file. Format inferred from output_path's
    extension (.png, .jpg, .bmp, .tiff, .webp, ...).

    Raises
    ------
    ImportError, InvalidImageError
    """
    if not PIL_AVAILABLE:
        raise ImportError("Install pillow: pip install pillow")
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    _validate_numpy(matrix)
    if matrix.dtype != np.uint8:
        matrix = (np.clip(matrix, 0, 1) * 255).astype(np.uint8)
    out_path = _validate_path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(matrix.squeeze()).save(out_path)
    logger.debug("Saved image to %s", out_path)


# ==================== Image Processing (point ops) ====================

def normalize(matrix: ImageArray) -> ImageArray:
    """Rescale values to [0, 1] using min-max normalization."""
    _validate_numpy(matrix)
    mat = matrix.astype(np.float32)
    lo, hi = mat.min(), mat.max()
    if hi - lo < 1e-12:
        return np.zeros_like(mat)
    return (mat - lo) / (hi - lo)


def invert(matrix: ImageArray) -> ImageArray:
    """Invert a normalized [0, 1] image matrix (1 - value)."""
    _validate_numpy(matrix)
    return 1.0 - matrix.astype(np.float32)


def adjust_brightness(matrix: ImageArray, delta: float) -> ImageArray:
    """Add `delta` to every pixel and clip to [0, 1]."""
    _validate_numpy(matrix)
    return np.clip(matrix.astype(np.float32) + delta, 0.0, 1.0)


def adjust_contrast(matrix: ImageArray, factor: float) -> ImageArray:
    """Scale contrast around 0.5 by `factor` and clip to [0, 1]."""
    _validate_numpy(matrix)
    mat = matrix.astype(np.float32)
    return np.clip((mat - 0.5) * factor + 0.5, 0.0, 1.0)


# ==================== Convolution Engine ====================

def convolve(matrix: ImageArray, kernel: np.ndarray, padding: PaddingMode = "reflect") -> ImageArray:
    """
    General-purpose 2D convolution, applied independently per channel.
    All filters below (blur, sharpen, edge detectors, SSIM) are built on
    top of this single vectorized primitive instead of duplicating loops.

    Parameters
    ----------
    matrix : np.ndarray
        Shape (H, W), (H, W, 1), (H, W, 3) or (H, W, 4).
    kernel : np.ndarray
        2D array with odd dimensions (e.g. 3x3, 5x5).
    padding : {"reflect", "constant", "edge", "symmetric"}, default "reflect"
        Border-handling mode, passed to np.pad.

    Returns
    -------
    np.ndarray of shape (H, W, C), dtype float32. Not clipped - callers
    that need a displayable image should clip/normalize the result.

    Raises
    ------
    InvalidShapeError
        If the kernel is not 2D with odd dimensions.

    Examples
    --------
    >>> img = np.ones((5, 5, 1), dtype=np.float32)
    >>> identity = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float32)
    >>> np.allclose(convolve(img, identity), img)
    True
    """
    mat = _ensure_channels(matrix).astype(np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    _validate_kernel(kernel)

    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(mat, ((ph, ph), (pw, pw), (0, 0)), mode=padding)
    windows = sliding_window_view(padded, (kh, kw), axis=(0, 1))  # (H, W, C, kh, kw)
    return np.einsum("hwckl,kl->hwc", windows, kernel)


def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Build a normalized 2D Gaussian kernel."""
    if size % 2 == 0:
        raise InvalidShapeError(f"Gaussian kernel size must be odd, got {size}")
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return (kernel / kernel.sum()).astype(np.float32)


def box_kernel(size: int = 3) -> np.ndarray:
    """Build a normalized box (mean) kernel."""
    if size % 2 == 0:
        raise InvalidShapeError(f"Box kernel size must be odd, got {size}")
    return np.ones((size, size), dtype=np.float32) / (size * size)


# ==================== Filters (built on convolve) ====================

def gaussian_blur(matrix: ImageArray, size: int = 5, sigma: float = 1.0) -> ImageArray:
    """Gaussian blur."""
    return convolve(matrix, gaussian_kernel(size, sigma))


def box_blur(matrix: ImageArray, size: int = 3) -> ImageArray:
    """Box (mean) blur."""
    return convolve(matrix, box_kernel(size))


def sharpen(matrix: ImageArray) -> ImageArray:
    """Sharpen filter, clipped to [0, 1]."""
    return np.clip(convolve(matrix, SHARPEN_KERNEL), 0.0, 1.0)


def emboss(matrix: ImageArray) -> ImageArray:
    """Emboss filter, clipped to [0, 1]."""
    return np.clip(convolve(matrix, EMBOSS_KERNEL), 0.0, 1.0)


def laplacian(matrix: ImageArray) -> ImageArray:
    """Laplacian edge-detection filter (unclipped)."""
    return convolve(matrix, LAPLACIAN_KERNEL)


def _gradient_magnitude(matrix: ImageArray, kx: np.ndarray, ky: np.ndarray) -> ImageArray:
    gx = convolve(matrix, kx)
    gy = convolve(matrix, ky)
    return normalize(np.sqrt(gx**2 + gy**2))


def sobel(matrix: ImageArray) -> ImageArray:
    """Sobel edge detector (gradient magnitude, normalized to [0, 1])."""
    return _gradient_magnitude(matrix, SOBEL_X_KERNEL, SOBEL_Y_KERNEL)


def prewitt(matrix: ImageArray) -> ImageArray:
    """Prewitt edge detector (gradient magnitude, normalized to [0, 1])."""
    return _gradient_magnitude(matrix, PREWITT_X_KERNEL, PREWITT_Y_KERNEL)


def roberts(matrix: ImageArray) -> ImageArray:
    """Roberts-cross edge detector (gradient magnitude, normalized to [0, 1])."""
    return _gradient_magnitude(matrix, ROBERTS_X_KERNEL, ROBERTS_Y_KERNEL)


# ==================== Histogram ====================

def histogram(matrix: ImageArray, bins: int = 256, value_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """
    Per-channel histogram.

    Returns
    -------
    np.ndarray of shape (C, bins)
    """
    mat = _ensure_channels(matrix)
    return np.stack(
        [np.histogram(mat[..., c], bins=bins, range=value_range)[0] for c in range(mat.shape[-1])]
    )


def cumulative_histogram(matrix: ImageArray, bins: int = 256, value_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Per-channel cumulative histogram, shape (C, bins)."""
    return histogram(matrix, bins, value_range).cumsum(axis=-1)


def equalize_histogram(matrix: ImageArray, bins: int = 256) -> ImageArray:
    """
    Histogram equalization per channel, for matrices with values in [0, 1].

    Returns
    -------
    np.ndarray, same shape as input, values in [0, 1].
    """
    mat = _ensure_channels(matrix).astype(np.float32)
    out = np.empty_like(mat)
    for c in range(mat.shape[-1]):
        channel = mat[..., c]
        hist, _ = np.histogram(channel, bins=bins, range=(0.0, 1.0))
        cdf = hist.cumsum().astype(np.float64)
        cdf = cdf / cdf[-1] if cdf[-1] > 0 else cdf
        idx = np.clip((channel * bins).astype(int), 0, bins - 1)
        out[..., c] = cdf[idx]
    return out.astype(np.float32)


# ==================== Color Space ====================

def rgb_to_gray(matrix: ImageArray) -> ImageArray:
    """RGB -> grayscale using ITU-R BT.709 luminosity weights."""
    mat = _ensure_channels(matrix)
    if mat.shape[-1] < 3:
        raise InvalidShapeError(f"Expected at least 3 channels (RGB), got {mat.shape}")
    gray = mat[..., 0] * 0.2126 + mat[..., 1] * 0.7152 + mat[..., 2] * 0.0722
    return gray[..., None].astype(np.float32)


def gray_to_rgb(matrix: ImageArray) -> ImageArray:
    """Grayscale -> RGB by repeating the single channel three times."""
    mat = _ensure_channels(matrix)
    if mat.shape[-1] != 1:
        raise InvalidShapeError(f"Expected a single-channel image, got {mat.shape}")
    return np.repeat(mat, 3, axis=-1)


def rgb_to_hsv(matrix: ImageArray) -> ImageArray:
    """RGB -> HSV. All channels (input and output) in [0, 1]."""
    mat = _ensure_channels(matrix)[..., :3].astype(np.float64)
    r, g, b = mat[..., 0], mat[..., 1], mat[..., 2]
    maxc = mat.max(axis=-1)
    minc = mat.min(axis=-1)
    delta = maxc - minc
    v = maxc
    s = np.where(maxc > 0, delta / np.where(maxc > 0, maxc, 1), 0.0)

    safe_delta = np.where(delta == 0, 1, delta)
    rc = (maxc - r) / safe_delta
    gc = (maxc - g) / safe_delta
    bc = (maxc - b) / safe_delta
    h = np.select(
        [maxc == r, maxc == g, maxc == b],
        [bc - gc, 2.0 + rc - bc, 4.0 + gc - rc],
        default=0.0,
    )
    h = np.where(delta == 0, 0.0, (h / 6.0) % 1.0)
    return np.stack([h, s, v], axis=-1).astype(np.float32)


def hsv_to_rgb(matrix: ImageArray) -> ImageArray:
    """HSV -> RGB. All channels (input and output) in [0, 1]."""
    mat = _ensure_channels(matrix)[..., :3].astype(np.float64)
    h, s, v = mat[..., 0] * 6.0, mat[..., 1], mat[..., 2]
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    conditions = [i == k for k in range(6)]
    r = np.select(conditions, [v, q, p, p, t, v], default=v)
    g = np.select(conditions, [t, v, v, q, p, p], default=v)
    b = np.select(conditions, [p, p, t, v, v, q], default=v)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def split_channels(matrix: ImageArray) -> Tuple[ImageArray, ...]:
    """Split into a tuple of single-channel (H, W, 1) arrays."""
    mat = _ensure_channels(matrix)
    return tuple(mat[..., c : c + 1] for c in range(mat.shape[-1]))


def merge_channels(*channels: ImageArray) -> ImageArray:
    """Merge single-channel arrays back into one (H, W, C) array."""
    if not channels:
        raise InvalidImageError("At least one channel is required")
    normalized = [_ensure_channels(ch) for ch in channels]
    return np.concatenate(normalized, axis=-1)


# ==================== Geometric Operations ====================

def crop(matrix: ImageArray, top: int, left: int, height: int, width: int) -> ImageArray:
    """Crop a (top, left, height, width) region out of the matrix."""
    mat = _ensure_channels(matrix)
    h, w = mat.shape[:2]
    if top < 0 or left < 0 or top + height > h or left + width > w:
        raise InvalidShapeError(
            f"Crop region out of bounds for image of shape {mat.shape}: "
            f"top={top}, left={left}, height={height}, width={width}"
        )
    return mat[top : top + height, left : left + width]


def flip_horizontal(matrix: ImageArray) -> ImageArray:
    """Flip left-right."""
    return _ensure_channels(matrix)[:, ::-1]


def flip_vertical(matrix: ImageArray) -> ImageArray:
    """Flip top-bottom."""
    return _ensure_channels(matrix)[::-1, :]


def pad(matrix: ImageArray, pad_width: int, mode: PaddingMode = "constant", constant_value: float = 0.0) -> ImageArray:
    """Pad the height/width dimensions by `pad_width` on every side."""
    mat = _ensure_channels(matrix)
    kwargs = {"constant_values": constant_value} if mode == "constant" else {}
    return np.pad(mat, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode=mode, **kwargs)


def rotate(matrix: ImageArray, angle: float) -> ImageArray:
    """
    Rotate by `angle` degrees (counter-clockwise).

    Multiples of 90 degrees are handled losslessly with np.rot90.
    Arbitrary angles require Pillow (for interpolation); without it, a
    NotImplementedError is raised rather 
