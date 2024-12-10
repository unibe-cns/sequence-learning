#!/usr/bin/env python3
from abc import ABC, abstractmethod
from typing import Tuple

import numba
import numpy as np
import numpy.typing as npt

from .config import NetworkConfig, NeuronConfig, WeightConfig
from .rate_buffer import Buffer

# disable numba jit for debugging etc.
numba.config.DISABLE_JIT = False


class Weights(ABC):
    """
    Base class for weight matrices in neural networks.
    """

    @abstractmethod
    def __init__(self, weight_params: WeightConfig):
        self.weight_matrix = None
        self.delays = None
        pass

    def __call__(self, num_vis: int, num_lat: int) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Create weight matrix.

        :param num_vis: Number of visible neurons.
        :type num_vis: int
        :param num_lat: Number of lateral neurons.
        :type num_lat: int
        :return: A tuple containing the created weight matrix and associated delays.
        :rtype: Tuple[npt.NDArray]
        """
        self.weight_matrix = self._create_weight_matrix(num_vis, num_lat)
        self.delays = self._create_delays(num_vis, num_lat)

        return self.weight_matrix, self.delays

    def _create_delays(self, num_vis: int, num_lat: int) -> Tuple[npt.NDArray]:
        "Create the delays associated with the weight matrix"
        """
        :param num_vis: Number of visible neurons.
        :type num_vis: int
        :param num_lat: Number of lateral neurons.
        :type num_lat: int
        :return: A tuple containing the created delays.
        :rtype: Tuple[npt.NDArray]
        """

        num_total = num_vis + num_lat
        d_min = self.delays[0]
        d_max = self.delays[1]
        delays = self.rng_d.integers(d_min, d_max, num_total)

        return delays

    @abstractmethod
    def _create_weight_matrix(self, num_vis: int, num_lat: int):
        pass


class DendriticWeights(Weights):
    """
    Class for creating dendritic weight matrices.

    This class extends the base Weights class and provides functionality to create
    dendritic weight matrices based on the given configuration.

    :param weight_params: Configuration object containing weight parameters.
    :type weight_params: WeightConfig
    """

    def __init__(self, weight_params: WeightConfig):
        """
        Initialize the DendriticWeights object.

        :param weight_params: Configuration object containing weight parameters.
        :type weight_params: WeightConfig
        """
        super().__init__(weight_params)
        self.delays = weight_params.d_den
        self.W_vis_vis = weight_params.W_vis_vis
        self.W_vis_lat = weight_params.W_vis_lat
        self.W_lat_vis = weight_params.W_lat_vis
        self.W_lat_lat = weight_params.W_lat_lat
        self.d_den = weight_params.d_den
        self.rng_w = np.random.default_rng(seed=weight_params.w_den_seed)
        self.rng_d = np.random.default_rng(seed=weight_params.d_den_seed)

    def _create_weight_matrix(self, num_vis: int, num_lat: int) -> Tuple[npt.NDArray]:
        """
        Create the dendritic weight matrix.

        This method initializes a weight matrix and fills it with random values
        based on the configuration parameters.

        :param num_vis: Number of visible neurons.
        :type num_vis: int
        :param num_lat: Number of lateral neurons.
        :type num_lat: int
        :return: The created weight matrix.
        :rtype: Tuple[npt.NDArray]
        """
        # Initialize weight matrix
        weights = np.zeros((num_vis + num_lat, num_vis + num_lat))
        weights[num_vis:, num_vis:] = self.rng_w.uniform(
            self.W_lat_lat[0], self.W_lat_lat[1], (num_lat, num_lat)
        )  # Lat to Lat
        weights[num_vis:, :num_vis] = self.rng_w.uniform(
            self.W_lat_vis[0], self.W_lat_vis[1], (num_lat, num_vis)
        )  # Lat to Vis
        weights[:num_vis, num_vis:] = self.rng_w.uniform(
            self.W_vis_lat[0], self.W_vis_lat[1], (num_vis, num_lat)
        )  # Vis to Lat
        weights[:num_vis, :num_vis:] = self.rng_w.uniform(
            self.W_vis_vis[0], self.W_vis_vis[1], (num_vis, num_vis)
        )  # Vis to Vis

        # remove self connections
        np.fill_diagonal(weights, 0)

        return weights


class SomaticWeights(Weights):
    """
    Class for creating somatic weight matrices.

    This class extends the base Weights class and provides functionality to create
    somatic weight matrices based on probabilistic connection rules.

    :param weight_params: Configuration object containing weight parameters.
    :type weight_params: WeightConfig
    """

    def __init__(self, weight_params: WeightConfig):
        """
        Initialize the SomaticWeights object.

        :param weight_params: Configuration object containing weight parameters.
        :type weight_params: WeightConfig
        """
        super().__init__(weight_params)
        self.delays = weight_params.d_som
        self.p = weight_params.p
        self.q = weight_params.q
        self.p0 = weight_params.p0
        self.p_first = 1 - self.p0
        self.rng_w = np.random.default_rng(seed=weight_params.w_som_seed)
        self.rng_d = np.random.default_rng(seed=weight_params.d_som_seed)
        self.inh_delay = weight_params.d_int

    def _create_weight_matrix(self, num_vis: int, num_lat: int) -> Tuple[npt.NDArray]:
        """
        Create the somatic weight matrix based on probabilistic connection rules.

        This method implements a complex algorithm to create connections between
        neurons based on various probabilities and rules.

        :param num_vis: Number of visible neurons.
        :type num_vis: int
        :param num_lat: Number of lateral neurons.
        :type num_lat: int
        :return: The created weight matrix.
        :rtype: Tuple[npt.NDArray]
        """
        num_total = num_vis + num_lat
        neurons_outgoing = np.arange(num_total)
        neurons_incoming = np.arange(num_vis, num_total)
        weight_matrix = np.zeros((num_total, num_total))

        # Incoming connections
        connections_in = np.zeros(num_total)
        connections_in[:num_vis] = 1

        # Outgoing connections
        connections_out = np.zeros(num_total)

        # Neurons that can make a connection
        neurons_unspent = np.arange(num_total)
        neurons_looking = neurons_outgoing[connections_in > 0]

        while len(neurons_looking) > 0:
            for idx_pre in neurons_looking:
                while np.isin(idx_pre, neurons_looking):
                    # Probability for forming connection
                    if connections_out[idx_pre] == 0:
                        prob_out = 1 - self.p0
                    else:
                        prob_out = np.power(self.p, connections_out[idx_pre])

                    # Test for formation
                    formation = self.rng_w.binomial(1, prob_out)
                    if not formation:
                        # Remove neuron pre from list of unspent neurons
                        neurons_unspent = neurons_unspent[neurons_unspent != idx_pre]
                        neurons_looking = np.delete(neurons_looking, 0)
                    else:
                        # Exclude connections to self and to neurons already connected
                        possible_post = neurons_incoming[neurons_incoming != idx_pre]
                        neurons_connected = np.where(weight_matrix[:, idx_pre] == 1)[0]
                        possible_post = np.setdiff1d(possible_post, neurons_connected)

                        formed = 0
                        while not formed:
                            post_idx = self.rng_w.choice(possible_post)
                            prob_in = np.power(self.q, connections_in[post_idx])
                            accept = self.rng_w.binomial(1, prob_in)
                            if accept:
                                # Add connection to matrix
                                weight_matrix[post_idx, idx_pre] = 1
                                connections_out[idx_pre] += 1
                                connections_in[post_idx] += 1
                                if np.isin(post_idx, neurons_unspent):
                                    neurons_unspent = neurons_unspent[
                                        neurons_unspent != post_idx
                                    ]
                                    neurons_looking = np.append(
                                        neurons_looking, post_idx
                                    )
                                formed = 1
                            else:
                                continue

        if np.sum(weight_matrix) != np.sum(connections_in) - num_vis:
            print("Problem with total connections")

        return weight_matrix

    def create_interneuron_delays(self) -> npt.NDArray:
        # Take self.delays and add the interneuron delay to all entries
        self.inh_delay = self.delays + self.inh_delay

        return self.inh_delay


class Network:
    def __init__(
        self,
        network_params: NetworkConfig,
        neuron_params: NeuronConfig,
        dendritic_weights: DendriticWeights,
        somatic_weights: SomaticWeights,
        rate_buffer: Buffer,
    ):
        self.neuron_params = neuron_params
        self.num_lat = network_params.num_lat
        self.num_vis = network_params.num_vis
        self.num_all = self.num_lat + self.num_vis

        self.dendritic_weights, self.dendritic_delays = dendritic_weights(
            self.num_lat, self.num_vis
        )
        self.somatic_weights, self.somatic_delays = somatic_weights(
            self.num_vis, self.num_lat
        )
        self.interneuron_delays = somatic_weights.create_interneuron_delays()

        self.r_rest = eq_phi(
            self.neuron_params.E_l, self.neuron_params.a, self.neuron_params.b
        )

        # Dynamical variables
        self.v = np.ones(self.num_all) * self.neuron_params.E_l
        self.u = np.ones(self.num_all) * self.neuron_params.E_l
        self.r_bar = np.ones(self.num_all) * self.r_rest
        self.r = np.ones(self.num_all) * self.r_rest
        self.r_den = np.ones(self.num_all) * self.r_rest
        self.r_exc = np.ones(self.num_all) * self.r_rest
        self.r_inh = np.ones(self.num_all) * self.r_rest

        self.dt = None

    def _compute_buffer_depth(self, dt):
        max_buffer_ms = max(max(self.dendritic_delays), max(self.interneuron_delays))
        buffer_depth = int(max_buffer_ms / dt)

        return buffer_depth

    def prepare_for_simulation(self, dt):
        # Configure rate buffer for simulation
        self.rate_buffer = Buffer(
            self.num_all, self._compute_buffer_depth(dt), self.r_rest
        )
        self.dt = dt

    def _compute_update(self, u_inp):
        # Compute delayed rates
        self.r_den = self.rate_buffer.get(self.dendritic_delays)
        self.r_exc = self.rate_buffer.get(self.somatic_delays)
        self.r_inh = self.rate_buffer.get(self.interneuron_delays)

        dudt, dvdt, dwdt, dr_bar_dt = total_diff_eq(
            u=self.u,
            v=self.v,
            w_den=self.dendritic_weights,
            r_bar=self.r_bar,
            r_den=self.r_den,
            r_exc=self.r_exc,
            r_inh=self.r_inh,
            u_inp=u_inp,
            w_som=self.somatic_weights,
            C_v=self.neuron_params.C_v,
            C_u=self.neuron_params.C_u,
            E_l=self.neuron_params.E_l,
            E_exc=self.neuron_params.E_exc,
            E_inh=self.neuron_params.E_inh,
            g_l=self.neuron_params.g_l,
            g_den=self.neuron_params.g_den,
            g_exc_0=self.neuron_params.g_exc_0,
            g_inh_0=self.neuron_params.g_inh_0,
            a=self.neuron_params.a,
            b=self.neuron_params.b,
            lam=self.neuron_params.lam,
        )

        return dudt, dvdt, dwdt, dr_bar_dt

    def _update_dyanmic_variables(self, dudt, dvdt, dwdt, dr_bar_dt, dt):
        self.u += dudt * dt
        self.v += dvdt * dt
        self.r_bar += dr_bar_dt * dt
        self.dendritic_weights += dwdt * dt

    def _update_rates_and_buffer(self, dt):
        new_r = eq_phi(self.u, self.a, self.b)
        self.r = new_r
        self.rate_buffer.roll(new_r)

    def simulation_step(self, dt, u_inp):
        dudt, dvdt, dwdt, dr_bar_dt = self._compute_update(dt, u_inp)
        self._update_dyanmic_variables(dudt, dvdt, dwdt, dr_bar_dt, dt)
        self._update_rates_and_buffer(dt)


#####################################
# equations of the dynamical system #
#####################################


@numba.njit
def eq_phi(u: npt.NDArray | float, a: float, b: float) -> npt.NDArray:
    """Activation function."""
    return 1.0 / (1 + np.exp(a * (b - u)))


@numba.njit()
def eq_i_den(r_delayed: npt.NDArray, w_den: npt.NDArray) -> npt.NDArray:
    """Dendritic synaptic current."""
    i_den = np.dot(w_den, r_delayed)
    return i_den


@numba.vectorize()
def eq_syn_cond(r_delay: float, g_0: float, phi_el: float) -> float:
    """Calc the synaptic conductances for the somatic weights."""
    if r_delay > phi_el:
        return r_delay * g_0
    else:
        return phi_el * g_0


@numba.njit()
def eq_rescale_v(v: npt.NDArray, g_l: float, g_den: float, E_l: float) -> npt.NDArray:
    """Rescale dendritic voltage to somatic for plasticity."""
    return (g_l * E_l + g_den * v) / (g_l + g_den)


@numba.njit()
def eq_drbar_dt(
    r_bar: npt.NDArray, r_delay: npt.NDArray, g_l: float, g_den: float
) -> npt.NDArray:
    """Diff. eq. for the lowpass filtered rate used in the plasticity."""
    return -g_l * r_bar + (g_l * g_den) / (g_l + g_den) * r_delay


@numba.njit()
def eq_i_som(
    w_som: npt.NDArray,
    g_exc: npt.NDArray,
    g_inh: npt.NDArray,
    u: npt.NDArray,
    E_exc: float,
    E_inh: float,
) -> npt.NDArray:
    """Calculate the somatic input current w/o teaching synapses."""
    i_som = np.dot(w_som, g_exc) * (E_exc - u) + np.dot(w_som, g_inh) * (E_inh - u)
    return i_som


@numba.njit()
def eq_cond_exc_inp(
    u_inp: npt.NDArray, lam: float, g_l: float, g_den: float, E_exc: float, E_inh: float
) -> npt.NDArray:
    """Calc the axcitatory conductances for the input."""
    g_exc = lam * (g_l + g_den) * (E_inh - u_inp) / ((1 - lam) * (E_inh - E_exc))
    return g_exc


@numba.njit()
def eq_cond_inh_inp(
    u_inp: npt.NDArray, lam: float, g_l: float, g_den: float, E_exc: float, E_inh: float
) -> npt.NDArray:
    """Calc the excitatory conductances for the input."""
    g_exc = lam * (g_l + g_den) * (u_inp - E_exc) / ((1 - lam) * (E_inh - E_exc))
    return g_exc


@numba.njit()
def eq_i_inp(
    g_exc_inp: npt.NDArray,
    g_inh_inp: npt.NDArray,
    u: npt.NDArray,
    E_exc: float,
    E_inh: float,
) -> npt.NDArray:
    """Calc the synaptic current of the input synapses.

    This is added only to the visible neurons.
    """
    i_inp = g_exc_inp * (E_exc - u) + g_inh_inp * (E_inh - u)
    return i_inp


@numba.njit()
def eq_dvdt(
    v: npt.NDArray, i_den: npt.NDArray, C_m: float, g_l: float, E_l: float
) -> npt.NDArray:
    """Diff. eq. for the dendritic potential v."""
    dvdt = (-g_l * (v - E_l) + i_den) / C_m
    return dvdt


@numba.njit()
def eq_dudt(
    u: npt.NDArray,
    v: npt.NDArray,
    i_som: npt.NDArray,
    C_m: float,
    g_l: float,
    g_den: float,
    E_l: float,
) -> npt.NDArray:
    """Diff. eq. for the somatic potential."""
    dudt = (-g_l * (u - E_l) + g_den * (v - u) + i_som) / C_m
    return dudt


@numba.njit()
def eq_dwdt(phi_u: npt.NDArray, phi_v: npt.NDArray, r_bar: npt.NDArray) -> npt.NDArray:
    """Urbanczik-Senn plasiticity rule."""
    dwdt = np.outer(phi_u - phi_v, r_bar)
    return dwdt


@numba.njit()
def total_diff_eq(
    u: npt.NDArray,
    v: npt.NDArray,
    w_den: npt.NDArray,
    r_bar: npt.NDArray,
    r_den: npt.NDArray,
    r_exc: npt.NDArray,
    r_inh: npt.NDArray,
    u_inp: npt.NDArray,
    w_som: npt.NDArray,
    C_u: float,
    C_v: float,
    g_l: float,
    g_den: float,
    g_exc_0: float,
    g_inh_0: float,
    E_l: float,
    E_exc: float,
    E_inh: float,
    a: float,
    b: float,
    lam: float,
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Calculate the total differential equations for a neural network model.

    This function computes the time derivatives of membrane potentials, synaptic
    weights, and firing rates for a network of neurons with dendritic and somatic
    compartments.

    :param u: Somatic membrane potentials, dynamic model variable (1-dim vector of size N)  # noqa
    :type u: numpy.ndarray
    :param v: Dendritic membrane potentials, dynamic model variable (1-dim vector of size N)  # noqa
    :type v: numpy.ndarray
    :param w_den: Dendritic synaptic weights, dynamic model variable (2-dim matrix of size NxN)  # noqa
    :type w_den: numpy.ndarray
    :param r_bar: Average firing rates, dynamic model variable (1-dim vector of size N)  # noqa
    :type r_bar: numpy.ndarray
    :param r_den: Delayed dendritic firing rates, instanteous variable (1-dim vector of size N)  # noqa
    :type r_den: numpy.ndarray
    :param r_exc: delayed somatic excitatory rates, instanteous variable (1-dim vector of size N)  # noqa
    :type r_exc: numpy.ndarray
    :param r_inh: delayed somatic inhibitory rates, instanteous variable (1-dim vector of size N)  # noqa
    :type r_inh: numpy.ndarray
    :param u_inp: Input membrane potentials, instanteous variable (1-dim vector of size N_visible)  # noqa
    :type u_inp: numpy.ndarray
    :param w_som: Sparse somatic synaptic weights, fixed model parameter (2-dim matrix of size NxN)  # noqa
    :type w_som: numpy.ndarray
    :param C_u: Somatic membrane capacitance, fixed model parameter
    :type C_u: float
    :param C_v: Dendritic membrane capacitance, fixed model parameter
    :type C_v: float
    :param g_l: Leak conductance, fixed model parameter
    :type g_l: float
    :param g_den: Dendritic coupling conductance, fixed model parameter
    :type g_den: float
    :param g_exc_0: Baseline excitatory conductance, fixed model parameter
    :type g_exc_0: float
    :param g_inh_0: Baseline inhibitory conductance, fixed model parameter
    :type g_inh_0: float
    :param E_l: Leak reversal potential, fixed model parameter
    :type E_l: float
    :param E_exc: Excitatory reversal potential, fixed model parameter
    :type E_exc: float
    :param E_inh: Inhibitory reversal potential, fixed model parameter
    :type E_inh: float
    :param a: Activation function parameter, fixed model parameter
    :type a: float
    :param b: Activation function parameter, fixed model parameter
    :type b: float
    :param lam: Input scaling factor, fixed model parameter
    :type lam: float

    :return: Tuple containing time derivatives of u, v, w, and r_bar
    :rtype: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]

    This function implements various equations for neural dynamics, including:
    - Dendritic and somatic currents
    - Synaptic conductances
    - Membrane potential dynamics
    - Firing rate dynamics
    - Synaptic plasticity

    The function uses helper functions (not shown here) to compute individual components
    of the model, such as eq_i_den, eq_syn_cond, eq_i_som, etc.
    """
    phi_el = eq_phi(E_l, a, b)
    num_vis = len(u_inp)

    # somato-dendritic connections
    i_den = eq_i_den(r_den, w_den)  # eq. 8

    # somato-somatic conentions
    g_exc = eq_syn_cond(r_exc, g_exc_0, phi_el)  # eq. 10
    g_inh = eq_syn_cond(r_inh, g_inh_0, phi_el)  # eq. 11
    i_som = eq_i_som(w_som, g_exc, g_inh, u, E_exc, E_inh)
    # input to visible layer
    g_exc_inp = eq_cond_exc_inp(u_inp, lam, g_l, g_den, E_exc, E_inh)
    g_inh_inp = eq_cond_inh_inp(u_inp, lam, g_l, g_den, E_exc, E_inh)
    i_som_inp = eq_i_inp(g_exc_inp, g_inh_inp, u[:num_vis], E_exc, E_inh)
    # total somatic input current:
    i_som[:num_vis] += i_som_inp

    # differential equations:
    dvdt = eq_dvdt(v, i_den, C_v, g_l, E_l)
    dudt = eq_dudt(u, v, i_som, C_u, g_l, g_den, E_l)
    dr_bar_dt = eq_drbar_dt(r_bar, r_den, g_l, g_den)
    # plasticity
    phi_u = eq_phi(u, a, b)
    v_rescaled = eq_rescale_v(v, g_l, g_den, E_l)
    phi_v = eq_phi(v_rescaled, a, b)
    dwdt = eq_dwdt(phi_u, phi_v, r_bar)

    return dudt, dvdt, dwdt, dr_bar_dt
