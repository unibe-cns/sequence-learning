#!/usr/bin/env python3

import pytest


@pytest.mark.parametrize(
    "a, b, expected", [(3, 5, 8), (2, 4, 6), (0, 0, 0), (-1, 1, 0), (10, -5, 5)]
)
def test_addition(a, b, expected):
    assert a + b == expected
