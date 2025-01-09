#!/usr/bin/env python3

import numpy as np
import pytest

from elise.stats import mse, pearson_coef, window_slider


@pytest.fixture
def sample_data():
    return np.random.rand(100, 5)


@pytest.fixture
def sample_target():
    return np.random.rand(10, 5)


def test_window_slider(sample_data, sample_target):
    def dummy_func(x, y):
        return np.mean(x - y)

    result = window_slider(sample_data, sample_target, dummy_func)
    assert isinstance(result, np.ndarray)
    assert result.shape == (91,)


def test_pearson_coef(sample_data, sample_target):
    result = pearson_coef(sample_data[:10], sample_target)
    assert isinstance(result, float)
    assert -1 <= result <= 1


def test_pearson_coef_perfect_correlation(sample_target):
    data = sample_target * -0.32
    result = pearson_coef(data, sample_target)
    assert result == pytest.approx(-1.0)


def test_mse(sample_data, sample_target):
    result = mse(sample_data[:10], sample_target)
    assert isinstance(result, float)
    assert result >= 0


def test_mse_perfect_values(sample_target):
    result = mse(sample_target, sample_target)
    assert result == pytest.approx(0.0)


def test_window_slider_with_pearson(sample_data, sample_target):
    result = window_slider(sample_data, sample_target, pearson_coef)
    assert isinstance(result, np.ndarray)
    assert result.shape == (91,)
    assert np.all((result >= -1) & (result <= 1))


def test_window_slider_with_mse(sample_data, sample_target):
    result = window_slider(sample_data, sample_target, mse)
    assert isinstance(result, np.ndarray)
    assert result.shape == (91,)
    assert np.all(result >= 0)
