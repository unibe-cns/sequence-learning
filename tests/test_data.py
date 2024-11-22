#!/usr/bin/env python3

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elise.data import Dataloader, OneHotPattern, Pattern

DT = 0.1

# TODO use the factory approach

###############################
# test the Base Pattern class #
###############################


@pytest.fixture()
def base_sequence():
    pattern = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return pattern


@pytest.fixture()
def pattern_factory(base_sequence):
    def factory(sequence=None, dt=None, duration=None):
        if sequence is None:
            sequence = base_sequence
        if dt is None and duration is None:
            dt = DT
        pattern = Pattern(sequence, dt=dt, duration=duration)
        return pattern

    return factory


@pytest.fixture()
def transform_plus():
    def plus(x, a=2.0):
        return x + a

    return plus


@pytest.fixture()
def transform_mult():
    def mult(x, m=2.0):
        return x * m

    return mult


@pytest.fixture()
def transform_reshape():
    def reshape(x):
        return np.expand_dims(x, -1)

    return reshape


class TestPattern:
    @pytest.mark.parametrize(("dt", "dur"), [(0.1, None), (None, 0.6)])
    def test_init(self, pattern_factory, base_sequence, dt, dur):
        pat = pattern_factory(dt=dt, duration=dur)
        assert pat.dt == pytest.approx(DT)
        assert pat.duration == pytest.approx(len(base_sequence) * DT)
        assert pat.shape == base_sequence.shape
        assert_allclose(base_sequence, pat.pattern)

    def test_init_error(self, pattern_factory):
        with pytest.raises(ValueError):
            _ = pattern_factory(dt=0.3, duration=0.7)

    def test_len(self, pattern_factory, base_sequence):
        pat = pattern_factory()
        assert len(pat) == base_sequence.shape[0]

    @pytest.mark.parametrize(
        "idx", [0, (2, 1), np.s_[4, 2], np.s_[:, 1], slice(0, 4, 2)]
    )
    def test_getitem(self, pattern_factory, base_sequence, idx):
        pat = pattern_factory()
        assert_allclose(pat[idx], base_sequence[idx])

    def test_transform(self, pattern_factory, base_sequence, transform_plus):
        pat = pattern_factory()
        pat.transform(transform_plus)
        expected = transform_plus(base_sequence)
        assert_allclose(pat.pattern, expected)

    def test_pretransform_reshape_error(self, pattern_factory, transform_reshape):
        pat = pattern_factory()
        with pytest.raises(ValueError) as exc_info:  # noqa
            pat.transform(transform_reshape)


################################
# Test the OneHotPattern class #
################################


@pytest.fixture()
def onehot_sequence():
    return np.array([1, 1, 0, 0, 2, 2])


@pytest.fixture()
def onehot_pattern(onehot_sequence):
    return OneHotPattern(onehot_sequence, dt=DT, width=3)


def test_onehot(onehot_pattern, base_sequence):
    assert_allclose(onehot_pattern[:, :], base_sequence)


###################
# Test Dataloader #
###################


@pytest.fixture()
def dataloader_factory(pattern_factory):
    def factory(pattern=None, pre_transforms=[], online_transforms=[]):
        if pattern is None:
            pattern = pattern_factory()
        return Dataloader(
            pattern, pre_transforms=pre_transforms, online_transforms=online_transforms
        )

    return factory


class TestDataloader:
    @pytest.mark.parametrize("transforms", [[], [transform_plus, transform_mult]])
    def test_init(self, dataloader_factory, base_sequence, transforms):
        dataloader = dataloader_factory(online_transforms=transforms)
        assert_allclose(dataloader.pattern[:, :], base_sequence)
        assert dataloader.dt == pytest.approx(DT)
        assert dataloader.online_transforms == transforms

    def test_pretransform_reshape_error(self, dataloader_factory, transform_reshape):
        with pytest.raises(ValueError) as exc_info:  # noqa
            dataloader = dataloader_factory(pre_transforms=[transform_reshape])  # noqa

    def test_pretransforms(
        self, dataloader_factory, transform_plus, transform_mult, base_sequence
    ):
        dataloader = dataloader_factory(pre_transforms=[transform_plus, transform_mult])
        expected = transform_mult(transform_plus(base_sequence))
        assert_allclose(dataloader.pattern, expected)

    @pytest.mark.parametrize(
        ("t", "idx"),
        [
            (0.1, 1),
            (0.5, 5),
            (0.6, 0),
            (0.7, 1),
            (1.1, -1),
            (1.2, 0),
            (0.05, 0),
            (0.15, 1),
            (0.55, 5),
            (0.61, 0),
        ],
    )
    def test_call(self, dataloader_factory, base_sequence, t, idx):
        dataloader = dataloader_factory()
        assert_allclose(dataloader(t), base_sequence[idx])

    @pytest.mark.parametrize(
        ("T", "idx", "dt"),
        [
            (0.5, 5, 0.1),
            (0.5, 5, 0.23),
            (0.5, 5, 0.01),
            (0.6, 0, 0.1),
            (0.6, 0, 0.23),
            (0.6, 0, 0.01),
            (1.1, -1, 0.1),
            (1.1, -1, 0.23),
            (1.1, -1, 0.01),
        ],
    )
    def test_call_consistency(self, dataloader_factory, base_sequence, T, idx, dt):
        dataloader = dataloader_factory()
        for t in np.arange(0.0, T, dt):
            _ = dataloader(t)
        assert_allclose(dataloader(T), base_sequence[idx])

    @pytest.mark.parametrize(
        ("T", "idx", "dt"),
        [
            (0.5, 5, 0.1),
            (0.6, 0, 0.1),
            (1.1, -1, 0.1),
        ],
    )
    def test_online_transforms(
        self,
        dataloader_factory,
        base_sequence,
        transform_plus,
        transform_mult,
        T,
        idx,
        dt,
    ):
        dataloader = dataloader_factory(
            online_transforms=[transform_plus, transform_mult]
        )
        for t in np.arange(0.0, T, dt):
            _ = dataloader(t)
        expected = transform_mult(transform_plus(base_sequence[idx]))
        assert_allclose(dataloader(T), expected)

    def test_iter(self, dataloader_factory, base_sequence):
        dataloader = dataloader_factory()
        idx = 0
        t_exp = 0.0
        for t, pat in dataloader.iter(0.0, 1.2, 0.1):
            assert t == t_exp
            assert_allclose(pat, base_sequence[idx % len(base_sequence)])
            t_exp += 0.1
            idx += 1
