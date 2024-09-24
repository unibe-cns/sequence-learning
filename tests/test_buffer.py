#!/usr/bin/env python3

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elise.rate_buffer import Buffer


@pytest.fixture
def sample_buffer():
    size, depth, val = 3, 5, 0.0
    return Buffer(size, depth, val)


@pytest.fixture
def sample_buffer_array():
    size, depth, val = 3, 5, np.arange(3)
    return Buffer(size, depth, val)


@pytest.fixture
def sample_values():
    return [np.array([float(i), float(i + 1), float(i + 2)]) for i in range(10)]


@pytest.fixture
def sample_delay():
    return np.array([2, 0, 1])


@pytest.fixture
def rolled_buffer_1(sample_buffer, sample_values):
    for i in range(1):
        sample_buffer.roll(sample_values[i])
    return sample_buffer


@pytest.fixture
def expected_rolled_1():
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    return expected


@pytest.fixture
def rolled_buffer_n(sample_buffer, sample_values):
    for i in range(7):
        sample_buffer.roll(sample_values[i])
    return sample_buffer


@pytest.fixture
def expected_rolled_n():
    expected = np.array(
        [
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ]
    )
    return expected


def test_buffer_init(sample_buffer_array):
    expected = np.array([[0.0, 1.0, 2.0]] * 5)
    assert_allclose(sample_buffer_array.buf, expected)


def test_roll_once(rolled_buffer_1, expected_rolled_1):
    assert_allclose(rolled_buffer_1.buf, expected_rolled_1)


def test_get_rolled_once(rolled_buffer_1, sample_delay, expected_rolled_1):
    expected = np.array([0.0, 1.0, 0.0])
    res = rolled_buffer_1.get(sample_delay)
    assert_allclose(res, expected)
    # check that get did not change the internal state!
    assert_allclose(rolled_buffer_1.buf, expected_rolled_1)


def test_getitem_rolled_once(rolled_buffer_1):
    res = rolled_buffer_1[-2:, 1:]
    expected = np.array([[0.0, 0.0], [1.0, 2.0]])
    assert_allclose(res, expected)


def test_roll_n(rolled_buffer_n, expected_rolled_n):
    assert_allclose(rolled_buffer_n.buf, expected_rolled_n)


def test_get_rolled_n(rolled_buffer_n, sample_delay, expected_rolled_n):
    expected = np.array([4.0, 7.0, 7.0])
    res = rolled_buffer_n.get(sample_delay)
    assert_allclose(res, expected)
    # check that get did not change the internal state!
    assert_allclose(rolled_buffer_n.buf, expected_rolled_n)


def test_getitem_rolled_n(rolled_buffer_n):
    res = rolled_buffer_n[-2:, 1:]
    expected = np.array([[6.0, 7.0], [7.0, 8.0]])
    assert_allclose(res, expected)
