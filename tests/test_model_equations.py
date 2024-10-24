#!/usr/bin/env python3


import numpy as np
import pytest
from numpy.testing import assert_allclose

from elise.config import NetworkConfig, NeuronConfig, WeightConfig
from elise.model import (
    eq_cond_exc_inp,
    eq_cond_inh_inp,
    eq_drbar_dt,
    eq_dudt,
    eq_dvdt,
    eq_dwdt,
    eq_i_den,
    eq_i_inp,
    eq_i_som,
    eq_phi,
    eq_rescale_v,
    eq_syn_cond,
    total_diff_eq,
)


@pytest.fixture
def std_nrn():
    return NeuronConfig()


@pytest.fixture
def std_wgt():
    return WeightConfig()


@pytest.fixture
def std_nw():
    return NetworkConfig(num_vis=2, num_lat=3)


@pytest.fixture
def sample_u():
    return np.array([-70.0, -65.0, -60.0, -55.0, -50.0])


@pytest.fixture
def sample_v():
    return np.array([-71.0, -66.0, -61.0, -56.0, -51.0])


@pytest.fixture
def sample_inp(std_nrn):
    # target pattern z = [0, 1]
    return np.array([std_nrn.E_l + 0.0, std_nrn.E_l + 20])


@pytest.fixture
def sample_r(sample_u, std_nrn):
    return eq_phi(sample_u, std_nrn.a, std_nrn.b)


@pytest.fixture
def sample_r_delay_den(sample_r):
    return sample_r * 0.99


@pytest.fixture
def sample_r_delay_exc(sample_r):
    return sample_r * 0.96


@pytest.fixture
def sample_r_delay_inh(sample_r):
    return sample_r * 0.92


@pytest.fixture
def sample_r_bar(sample_r):
    return sample_r * 0.94


@pytest.fixture
def sample_g_exc(sample_r, std_nrn):
    phi_el = eq_phi(std_nrn.E_l, std_nrn.a, std_nrn.b)
    g_exc = eq_syn_cond(sample_r, std_nrn.g_exc_0, phi_el)
    return g_exc


@pytest.fixture
def sample_g_inh(sample_r, std_nrn):
    phi_el = eq_phi(std_nrn.E_l, std_nrn.a, std_nrn.b)
    g_inh = eq_syn_cond(sample_r * 0.95, std_nrn.g_inh_0, phi_el)
    return g_inh


@pytest.fixture
def sample_g_exc_inp(sample_inp, std_nrn):
    res = eq_cond_exc_inp(
        sample_inp,
        std_nrn.lam,
        std_nrn.g_l,
        std_nrn.g_den,
        std_nrn.E_exc,
        std_nrn.E_inh,
    )
    return res


@pytest.fixture
def sample_g_inh_inp(sample_inp, std_nrn):
    res = eq_cond_inh_inp(
        sample_inp,
        std_nrn.lam,
        std_nrn.g_l,
        std_nrn.g_den,
        std_nrn.E_exc,
        std_nrn.E_inh,
    )
    return res


@pytest.fixture
def sample_W_den(std_nw):
    """Sets up the dendritic weight matrix.

    Not produces by the actual weight creation class.
    """
    size = std_nw.num_vis + std_nw.num_lat
    w = np.linspace(0.0, 0.5, size**2).reshape(size, size)
    np.fill_diagonal(w, 0.0)
    return w


@pytest.fixture
def sample_W_som():
    w = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
        ]
    )
    return w


@pytest.fixture
def sample_i_den(sample_r, sample_W_den):
    res = eq_i_den(sample_r, sample_W_den)
    return res


@pytest.fixture
def sample_i_som(sample_W_som, sample_g_exc, sample_g_inh, sample_u, std_nrn):
    res = eq_i_som(
        sample_W_som, sample_g_exc, sample_g_inh, sample_u, std_nrn.E_exc, std_nrn.E_inh
    )
    return res


@pytest.fixture
def sample_derivatives(
    sample_u,
    sample_v,
    sample_r_delay_den,
    sample_r_delay_exc,
    sample_r_delay_inh,
    sample_r_bar,
    sample_inp,
    sample_W_som,
    sample_W_den,
    std_nrn,
    std_nw,
):
    """reference implementation for total_diff_eq.

    No guarantee that this is actually correct...
    However, if this and total_diff_eq yield the same results, the probabilty for
    for correctness is higher.
    Should be quite handy to have if we change implementation.
    """
    i_den = eq_i_den(sample_r_delay_den, sample_W_den)

    phi_el = eq_phi(std_nrn.E_l, std_nrn.a, std_nrn.b)

    g_exc = eq_syn_cond(sample_r_delay_exc, std_nrn.g_exc_0, phi_el)
    g_inh = eq_syn_cond(sample_r_delay_inh, std_nrn.g_inh_0, phi_el)

    i_som = eq_i_som(sample_W_som, g_exc, g_inh, sample_u, std_nrn.E_exc, std_nrn.E_inh)
    g_exc_inp = eq_cond_exc_inp(
        sample_inp,
        std_nrn.lam,
        std_nrn.g_l,
        std_nrn.g_den,
        std_nrn.E_exc,
        std_nrn.E_inh,
    )
    g_inh_inp = eq_cond_inh_inp(
        sample_inp,
        std_nrn.lam,
        std_nrn.g_l,
        std_nrn.g_den,
        std_nrn.E_exc,
        std_nrn.E_inh,
    )
    i_som_inp = eq_i_inp(
        g_exc_inp, g_inh_inp, sample_u[: std_nw.num_vis], std_nrn.E_exc, std_nrn.E_inh
    )
    i_som[: std_nw.num_vis] += i_som_inp

    dvdt = eq_dvdt(sample_v, i_den, std_nrn.C_v, std_nrn.g_l, std_nrn.E_l)
    dudt = eq_dudt(
        sample_u, sample_v, i_som, std_nrn.C_u, std_nrn.g_l, std_nrn.g_den, std_nrn.E_l
    )

    # input to plasticity rule
    phi_u = eq_phi(sample_u, std_nrn.a, std_nrn.b)
    v_rescaled = eq_rescale_v(sample_v, std_nrn.g_l, std_nrn.g_den, std_nrn.E_l)
    phi_v_rescaled = eq_phi(v_rescaled, std_nrn.a, std_nrn.b)
    dwdt = eq_dwdt(phi_u, phi_v_rescaled, sample_r_bar)
    dr_bar_dt = eq_drbar_dt(
        sample_r_bar, sample_r_delay_den, std_nrn.g_l, std_nrn.g_den
    )

    return dudt, dvdt, dwdt, dr_bar_dt


def test_activation_function(sample_u, std_nrn):
    """Tests the activation function"""
    res = eq_phi(sample_u, std_nrn.a, std_nrn.b)
    expected = np.array([0.02659699, 0.10909682, 0.35434369, 0.7109495, 0.9168273])
    assert_allclose(res, expected, rtol=1e-6)


def test_i_den(sample_r, sample_W_den, sample_i_den):
    """Tests the Dentritic synaptic input current.

    I_den = W_den @ r_delayed
    """
    expected = np.array([0.13787379, 0.34484235, 0.49049918, 0.5330847, 0.56188277])
    assert_allclose(sample_i_den, expected, rtol=1e-6)


def test_syn_cond(sample_r, std_nrn, sample_g_inh, sample_g_exc):
    """Tests the somatic conductances.

    eq. 10 and 11 in the manuscript
    """
    phi_el = eq_phi(std_nrn.E_l, std_nrn.a, std_nrn.b)

    # test for excitatory synapses
    expected = std_nrn.g_exc_0 * sample_r
    expected[sample_r <= phi_el] = phi_el * std_nrn.g_exc_0
    assert_allclose(sample_g_exc, expected)

    # test for inhibitory synapses
    expected = std_nrn.g_inh_0 * sample_r * 0.95
    expected[sample_r * 0.95 <= phi_el] = phi_el * std_nrn.g_inh_0
    assert_allclose(sample_g_inh, expected)


def test_rescale_v(sample_u, std_nrn):
    """Tests the computation of the rescaled dendritic voltage.

    Needed in the plasticity rule.
    """
    res = eq_rescale_v(sample_u, std_nrn.g_l, std_nrn.g_den, std_nrn.E_l)
    expected = (
        1.0
        / (std_nrn.g_l + std_nrn.g_den)
        * (std_nrn.g_l * std_nrn.E_l + std_nrn.g_den * sample_u)
    )
    assert_allclose(res, expected)


def test_drbar_dt(sample_r, std_nrn):
    """Tests low pass filtered rate."""
    sample_rbar = sample_r * 1.2
    res = eq_drbar_dt(sample_rbar, sample_r, std_nrn.g_l, std_nrn.g_den)
    expected = (
        -std_nrn.g_l * sample_rbar
        + (std_nrn.g_l * std_nrn.g_den) / (std_nrn.g_l + std_nrn.g_den) * sample_r
    )
    assert_allclose(res, expected)


def test_i_som(
    sample_u, std_nrn, sample_g_exc, sample_g_inh, sample_W_som, sample_i_som
):
    """Tests the somatic input current without external teaching."""

    expected = np.zeros(sample_u.shape)
    for j, u in enumerate(sample_u):
        for i, (g_exc, g_inh) in enumerate(zip(sample_g_exc, sample_g_inh)):
            expected[j] += sample_W_som[j, i] * (
                g_exc * (std_nrn.E_exc - u) + g_inh * (std_nrn.E_inh - u)
            )

    assert_allclose(sample_i_som, expected)


def test_inp_cond(sample_inp, sample_g_exc_inp, sample_g_inh_inp, std_nrn):
    """Test the target input synaptic conductances"""
    # excitatory
    expected = (
        std_nrn.lam
        / (1 - std_nrn.lam)
        * (std_nrn.g_l + std_nrn.g_den)
        * (std_nrn.E_inh - sample_inp)
        / (std_nrn.E_inh - std_nrn.E_exc)
    )
    assert_allclose(sample_g_exc_inp, expected)
    # inhibitory
    expected = (
        std_nrn.lam
        / (1 - std_nrn.lam)
        * (std_nrn.g_l + std_nrn.g_den)
        * (sample_inp - std_nrn.E_exc)
        / (std_nrn.E_inh - std_nrn.E_exc)
    )
    assert_allclose(sample_g_inh_inp, expected)


def test_i_inp(sample_u, sample_g_exc_inp, sample_g_inh_inp, std_nrn, std_nw):
    """Test the input currents of the target synapses.

    This current is added to the visible somas only.
    """
    u = sample_u[: std_nw.num_vis]
    res = eq_i_inp(sample_g_exc_inp, sample_g_inh_inp, u, std_nrn.E_exc, std_nrn.E_inh)

    expected = np.zeros_like(u)
    for i, (u_in, g_exc, g_inh) in enumerate(
        zip(u, sample_g_exc_inp, sample_g_inh_inp)
    ):
        expected[i] = g_exc * (std_nrn.E_exc - u_in) + g_inh * (std_nrn.E_inh - u_in)
    assert_allclose(res, expected)


def test_dvdt(sample_v, sample_i_den, std_nrn):
    """Tests the differential equation for the dendritiv voltage."""
    res = eq_dvdt(sample_v, sample_i_den, std_nrn.C_v, std_nrn.g_l, std_nrn.E_l)
    expected = np.zeros_like(sample_v)

    for i, (v, i_den) in enumerate(zip(sample_v, sample_i_den)):
        expected[i] = (-std_nrn.g_l * (v - std_nrn.E_l) + i_den) / std_nrn.C_v
    assert_allclose(res, expected)


def test_dudt(sample_u, sample_v, sample_i_som, std_nrn):
    """Tests the diff. eq. for the somatic potential."""
    res = eq_dudt(
        sample_u,
        sample_v,
        sample_i_som,
        std_nrn.C_u,
        std_nrn.g_l,
        std_nrn.g_den,
        std_nrn.E_l,
    )

    expected = np.zeros_like(sample_v)

    for i, (u, v, i_som) in enumerate(zip(sample_u, sample_v, sample_i_som)):
        expected[i] = (
            -std_nrn.g_l * (u - std_nrn.E_l) + std_nrn.g_den * (v - u) + i_som
        ) / std_nrn.C_u
    assert_allclose(res, expected)


def test_dwdt(sample_u, sample_v, sample_r, std_nrn, std_nw):
    """Tests plasticity rule for the dendritic weights."""
    phi_u = eq_phi(sample_u, std_nrn.a, std_nrn.b)
    v_rescaled = eq_rescale_v(sample_v, std_nrn.g_l, std_nrn.g_den, std_nrn.E_l)
    phi_v = eq_phi(v_rescaled, std_nrn.a, std_nrn.b)
    res = eq_dwdt(phi_u, phi_v, sample_r)

    n_tot = std_nw.num_vis + std_nw.num_lat
    expected = np.zeros((n_tot, n_tot))
    for i in range(n_tot):
        for j in range(n_tot):
            expected[j, i] = (phi_u[j] - phi_v[j]) * sample_r[i]

    assert_allclose(res, expected)


def test_total_diff_eq(
    sample_u,
    sample_v,
    sample_W_den,
    sample_r_bar,
    sample_r_delay_den,
    sample_r_delay_exc,
    sample_r_delay_inh,
    sample_inp,
    sample_W_som,
    std_nrn,
    sample_derivatives,
):
    """Tests the big function total_diff_eq."""
    dudt, dvdt, dwdt, dr_bar_dt = total_diff_eq(
        u=sample_u,
        v=sample_v,
        w_den=sample_W_den,
        r_bar=sample_r_bar,
        r_den=sample_r_delay_den,
        r_exc=sample_r_delay_exc,
        r_inh=sample_r_delay_inh,
        u_inp=sample_inp,
        w_som=sample_W_som,
        C_u=std_nrn.C_u,
        C_v=std_nrn.C_v,
        g_l=std_nrn.g_l,
        g_den=std_nrn.g_den,
        g_exc_0=std_nrn.g_exc_0,
        g_inh_0=std_nrn.g_inh_0,
        E_l=std_nrn.E_l,
        E_exc=std_nrn.E_exc,
        E_inh=std_nrn.E_inh,
        a=std_nrn.a,
        b=std_nrn.b,
        lam=std_nrn.lam,
    )

    expected = sample_derivatives

    assert_allclose(dudt, expected[0])
    assert_allclose(dvdt, expected[1])
    assert_allclose(dwdt, expected[2])
    assert_allclose(dr_bar_dt, expected[3])
