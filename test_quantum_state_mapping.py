"""
Unit tests for quantum_state_mapping.py
Run with: pytest test_quantum_state_mapping.py -v
"""
import numpy as np
import pytest

from quantum_state_mapping import (
    STATE_DIM,
    calculate_probabilities,
    calculate_probability_0,
    calculate_probability_1,
    image_to_qubit_states,
    is_normalized,
    pixel_to_phase_state,
    pixel_to_qubit_state,
)


# ---------- pixel_to_qubit_state (angle encoding) ----------

def test_zero_pixel_is_pure_zero_state():
    state = pixel_to_qubit_state(0.0)
    assert np.allclose(state, [1.0, 0.0])


def test_one_pixel_is_pure_one_state():
    state = pixel_to_qubit_state(1.0)
    assert np.allclose(state, [0.0, 1.0], atol=1e-6)


def test_half_pixel_is_equal_superposition():
    state = pixel_to_qubit_state(0.5)
    assert np.allclose(state, [1 / np.sqrt(2), 1 / np.sqrt(2)], atol=1e-6)


def test_angle_encoding_is_always_normalized():
    values = np.linspace(0, 1, 50)
    states = pixel_to_qubit_state(values)
    assert is_normalized(states)


def test_out_of_range_value_is_clipped_and_warns():
    with pytest.warns(UserWarning):
        state = pixel_to_qubit_state(1.5)
    assert np.allclose(state, pixel_to_qubit_state(1.0), atol=1e-6)


def test_negative_value_is_clipped_and_warns():
    with pytest.warns(UserWarning):
        state = pixel_to_qubit_state(-0.5)
    assert np.allclose(state, pixel_to_qubit_state(0.0), atol=1e-6)


def test_warn_on_clip_false_suppresses_warning(recwarn):
    pixel_to_qubit_state(1.5, warn_on_clip=False)
    assert len(recwarn) == 0


def test_non_numeric_input_raises_type_error():
    with pytest.raises(TypeError):
        pixel_to_qubit_state("not a number")


def test_complex_input_raises_type_error():
    with pytest.raises(TypeError):
        pixel_to_qubit_state(1 + 2j)


def test_batch_input_shape():
    values = np.array([0.0, 0.5, 1.0])
    states = pixel_to_qubit_state(values)
    assert states.shape == (3, STATE_DIM)


def test_image_shaped_input():
    image = np.random.rand(4, 5)
    states = pixel_to_qubit_state(image)
    assert states.shape == (4, 5, STATE_DIM)


# ---------- pixel_to_phase_state ----------

def test_phase_state_always_equal_probability():
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        state = pixel_to_phase_state(value)
        p0, p1 = calculate_probabilities(state)
        assert p0 == pytest.approx(0.5, abs=1e-6)
        assert p1 == pytest.approx(0.5, abs=1e-6)


def test_phase_state_is_normalized():
    values = np.linspace(0, 1, 20)
    states = pixel_to_phase_state(values)
    assert is_normalized(states)


def test_phase_state_dtype_is_complex():
    state = pixel_to_phase_state(0.5)
    assert np.iscomplexobj(state)


def test_phase_state_zero_pixel_has_zero_phase():
    state = pixel_to_phase_state(0.0)
    # phi=0 => second amplitude has zero imaginary part
    assert np.isclose(state[1].imag, 0.0, atol=1e-6)


def test_phase_encoding_rejects_complex_input():
    with pytest.raises(TypeError):
        pixel_to_phase_state(1 + 2j)


# ---------- image_to_qubit_states ----------

def test_image_to_qubit_states_requires_ndarray():
    with pytest.raises(TypeError):
        image_to_qubit_states([[0.0, 1.0]])  # plain list, not ndarray


def test_image_to_qubit_states_rejects_empty():
    with pytest.raises(ValueError):
        image_to_qubit_states(np.array([]))


def test_image_to_qubit_states_matches_pixel_function():
    image = np.array([[0.0, 1.0], [0.5, 0.25]])
    assert np.allclose(image_to_qubit_states(image), pixel_to_qubit_state(image))


# ---------- is_normalized ----------

def test_is_normalized_true_for_valid_state():
    assert is_normalized(np.array([1.0, 0.0]))


def test_is_normalized_false_for_invalid_state():
    assert not is_normalized(np.array([1.0, 1.0]))


def test_is_normalized_rejects_wrong_shape():
    with pytest.raises(ValueError):
        is_normalized(np.array([1.0, 0.0, 0.0]))


def test_is_normalized_works_on_batches():
    states = pixel_to_qubit_state(np.linspace(0, 1, 10))
    assert is_normalized(states)


# ---------- probability functions ----------

def test_probabilities_sum_to_one():
    state = pixel_to_qubit_state(0.3)
    p0, p1 = calculate_probabilities(state)
    assert p0 + p1 == pytest.approx(1.0, abs=1e-6)


def test_probability_functions_reject_wrong_shape():
    bad = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        calculate_probability_0(bad)
    with pytest.raises(ValueError):
        calculate_probability_1(bad)


def test_probability_scalar_return_type():
    state = pixel_to_qubit_state(0.5)
    p0 = calculate_probability_0(state)
    assert isinstance(p0, float)


def test_probability_batch_return_type():
    states = pixel_to_qubit_state(np.array([0.0, 0.5, 1.0]))
    p0 = calculate_probability_0(states)
    assert isinstance(p0, np.ndarray)
    assert p0.shape == (3,)
