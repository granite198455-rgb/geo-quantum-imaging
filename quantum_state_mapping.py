"""
Quantum State Mapping (Phase 2) - Improved & Vectorized
==========================================================
Maps normalized pixel values [0, 1] to 2D quantum state vectors (qubits).

Two encodings are provided:

1. Angle Encoding (real-valued amplitudes):
   |psi> = cos(theta/2)|0> + sin(theta/2)|1>,  theta = pixel_value * pi

2. Phase Encoding (complex amplitudes):
   |psi> = (|0> + e^(i*phi)|1>) / sqrt(2),  phi = pixel_value * 2*pi

Both support a single pixel (float) or an entire image matrix
(np.ndarray of any shape), so they compose with image_matrix_utils.py.

Scope note: this module is intended for classical feature encoding and
educational simulation, not full quantum circuit simulation. It does not
model entanglement, multi-qubit registers, gates, or measurement
collapse - for that, use a real framework such as Qiskit or Cirq.
"""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np

ArrayOrFloat = Union[float, np.ndarray]

# --- Constants ---
STATE_DIM = 2  # |0>, |1>
NORMALIZATION_ATOL = 1e-5

__all__ = [
    "pixel_to_qubit_state",
    "pixel_to_phase_state",
    "image_to_qubit_states",
    "is_normalized",
    "calculate_probability_0",
    "calculate_probability_1",
    "calculate_probabilities",
]


def _to_real_array(pixel_value: ArrayOrFloat) -> np.ndarray:
    """Shared input validation for the real-valued (angle) encoding path."""
    if np.iscomplexobj(pixel_value):
        raise TypeError("Complex numbers are not supported by pixel_to_qubit_state (angle encoding).")
    try:
        return np.asarray(pixel_value, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"pixel_value must be numeric (float or array), got {type(pixel_value).__name__}"
        ) from e


def pixel_to_qubit_state(pixel_value: ArrayOrFloat, warn_on_clip: bool = True) -> np.ndarray:
    """
    Map pixel value(s) in [0, 1] to quantum state vector(s) via angle encoding.

    Parameters
    ----------
    pixel_value : float or np.ndarray
        A single pixel value, or an array of any shape (e.g. a full image
        matrix). Values outside [0, 1] are clipped.
    warn_on_clip : bool, default True
        If True, emit a warning when input values needed clipping (this
        usually indicates upstream data wasn't normalized).

    Returns
    -------
    np.ndarray
        Shape (..., 2): the last axis holds [amplitude_0, amplitude_1].
        For a scalar input, shape is (2,). For an (H, W) image, shape is
        (H, W, 2).

    Raises
    ------
    TypeError
        If pixel_value is not numeric, or is complex.

    Examples
    --------
    >>> pixel_to_qubit_state(0.0)
    array([1., 0.], dtype=float32)
    >>> pixel_to_qubit_state(np.array([0.0, 1.0])).shape
    (2, 2)
    """
    arr = _to_real_array(pixel_value)
    clipped = np.clip(arr, 0.0, 1.0)
    if warn_on_clip and not np.allclose(arr, clipped):
        warnings.warn(
            "pixel_to_qubit_state: input contained values outside [0, 1]; they were clipped.",
            stacklevel=2,
        )

    theta = clipped * np.pi
    amplitude_0 = np.cos(theta / 2.0)
    amplitude_1 = np.sin(theta / 2.0)
    return np.stack([amplitude_0, amplitude_1], axis=-1).astype(np.float32)


def pixel_to_phase_state(pixel_value: ArrayOrFloat, warn_on_clip: bool = True) -> np.ndarray:
    """
    Map pixel value(s) in [0, 1] to quantum state vector(s) via phase
    encoding: |psi> = (|0> + e^(i*phi)|1>) / sqrt(2), phi = pixel_value * 2*pi.

    Unlike pixel_to_qubit_state, |alpha| and |beta| are always equal
    (both 1/sqrt(2)); the pixel value is encoded in the *relative phase*
    of |1> rather than in the split of probability between |0> and |1>.
    This means calculate_probability_0/1 will always report 0.5/0.5 for
    phase-encoded states - phase information isn't recoverable from a
    single-qubit measurement probability, only via interference with
    another state. This is expected, not a bug.

    Parameters
    ----------
    pixel_value : float or np.ndarray
    warn_on_clip : bool, default True

    Returns
    -------
    np.ndarray (complex128)
        Shape (..., 2).

    Raises
    ------
    TypeError
        If pixel_value is not numeric, or is complex.

    Examples
    --------
    >>> state = pixel_to_phase_state(0.0)
    >>> np.allclose(state, [1/np.sqrt(2), 1/np.sqrt(2)])
    True
    """
    arr = _to_real_array(pixel_value)
    clipped = np.clip(arr, 0.0, 1.0)
    if warn_on_clip and not np.allclose(arr, clipped):
        warnings.warn(
            "pixel_to_phase_state: input contained values outside [0, 1]; they were clipped.",
            stacklevel=2,
        )

    phi = clipped * 2.0 * np.pi
    amplitude_0 = np.full_like(phi, 1.0 / np.sqrt(2), dtype=np.complex128)
    amplitude_1 = (1.0 / np.sqrt(2)) * np.exp(1j * phi)
    return np.stack([amplitude_0, amplitude_1], axis=-1)


def image_to_qubit_states(image_matrix: np.ndarray, warn_on_clip: bool = True) -> np.ndarray:
    """
    Convenience wrapper: map every pixel of an image matrix to a qubit
    state (angle encoding). Equivalent to pixel_to_qubit_state but named
    for clarity when the input is a full image (e.g. from
    image_matrix_utils.normalize()).

    Parameters
    ----------
    image_matrix : np.ndarray
        Shape (H, W) or (H, W, C), values expected in [0, 1].

    Returns
    -------
    np.ndarray
        Shape (H, W, 2) or (H, W, C, 2).
    """
    if not isinstance(image_matrix, np.ndarray):
        raise TypeError(f"image_matrix must be an np.ndarray, got {type(image_matrix).__name__}")
    if image_matrix.size == 0:
        raise ValueError("image_matrix is empty")
    return pixel_to_qubit_state(image_matrix, warn_on_clip=warn_on_clip)


def _validate_state_shape(state_vector: np.ndarray) -> np.ndarray:
    state_vector = np.asarray(state_vector)
    if state_vector.ndim == 0 or state_vector.shape[-1] != STATE_DIM:
        raise ValueError(
            f"Expected state vector(s) with last dimension {STATE_DIM}, got shape {state_vector.shape}"
        )
    return state_vector


def is_normalized(state_vector: np.ndarray, atol: float = NORMALIZATION_ATOL) -> bool:
    """
    Check that |alpha|^2 + |beta|^2 == 1 (within tolerance) along the last
    axis. Works for a single state (shape (2,)) or a batch (shape (...,2)),
    and for both real and complex amplitudes.

    Raises
    ------
    ValueError
        If the input's last dimension isn't STATE_DIM (2).
    """
    state_vector = _validate_state_shape(state_vector)
    total_prob = np.sum(np.abs(state_vector) ** 2, axis=-1)
    return bool(np.allclose(total_prob, 1.0, atol=atol))


def calculate_probability_0(state_vector: np.ndarray) -> ArrayOrFloat:
    """Probability of measuring |0>: |alpha|^2, taken along the last axis."""
    state_vector = _validate_state_shape(state_vector)
    result = np.abs(state_vector[..., 0]) ** 2
    return float(result) if result.ndim == 0 else result


def calculate_probability_1(state_vector: np.ndarray) -> ArrayOrFloat:
    """Probability of measuring |1>: |beta|^2, taken along the last axis."""
    state_vector = _validate_state_shape(state_vector)
    result = np.abs(state_vector[..., 1]) ** 2
    return float(result) if result.ndim == 0 else result


def calculate_probabilities(state_vector: np.ndarray) -> tuple[ArrayOrFloat, ArrayOrFloat]:
    """Return (P(|0>), P(|1>)) together, guaranteed to sum to 1."""
    return calculate_probability_0(state_vector), calculate_probability_1(state_vector)


# ==================== Demo ====================
if __name__ == "__main__":
    print("--- Phase 2: Quantum State Mapping Simulation (Improved) ---")

    test_pixels = {
        "Dark Pixel (e.g., Background/Noise)": 0.0,
        "Mid-Tone Pixel (Anomalous Boundary)": 0.5,
        "Bright Pixel (e.g., Mineral Vein/Dense Tissue)": 1.0,
    }

    for description, value in test_pixels.items():
        state = pixel_to_qubit_state(value)
        prob_0, prob_1 = calculate_probabilities(state)
        print(f"\nTarget: {description} [Value: {value}]")
        print(f"  Quantum State Vector [|0>, |1>]: {state}")
        print(f"  Probability of Measuring |0>: {prob_0:.2f}")
        print(f"  Probability of Measuring |1>: {prob_1:.2f}")
        print(f"  Normalized: {is_normalized(state)}")

    print("\n--- Vectorized / Batch Demo (whole image at once) ---")
    fake_image = np.array([[0.0, 0.25], [0.5, 1.0]])  # a tiny 2x2 "image"
    states = image_to_qubit_states(fake_image)
    print("Image shape:", fake_image.shape, "-> States shape:", states.shape)
    p0, p1 = calculate_probabilities(states)
    print("P(|0>) per pixel:\n", p0)
    print("P(|1>) per pixel:\n", p1)
    print("All normalized:", is_normalized(states))

    print("\n--- Phase Encoding Demo ---")
    for value in (0.0, 0.25, 0.5, 1.0):
        pstate = pixel_to_phase_state(value)
        print(f"  value={value}: state={pstate}, "
              f"P(|0>)={calculate_probability_0(pstate):.2f}, "
              f"normalized={is_normalized(pstate)}")

    print("\n--- Error / Edge Case Handling ---")
    try:
        pixel_to_qubit_state("not a number")
    except TypeError as e:
        print(f"TypeError correctly raised: {e}")

    try:
        pixel_to_qubit_state(1 + 2j)
    except TypeError as e:
        print(f"TypeError correctly raised for complex input: {e}")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pixel_to_qubit_state(1.5)
        print(f"Clipping warning correctly raised: {w[0].message}")

    try:
        calculate_probability_0(np.array([1.0, 0.0, 0.0]))
    except ValueError as e:
        print(f"ValueError correctly raised for bad shape: {e}")

    try:
        is_normalized(np.array([1.0, 2.0, 3.0]))
    except ValueError as e:
        print(f"ValueError correctly raised in is_normalized for bad shape: {e}")

    print("\nFile ready! Save it as quantum_state_mapping.py")
