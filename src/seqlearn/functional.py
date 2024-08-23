#!/usr/bin/env python3

"""
Timo's proposal for a fully functional approach.

This is really just meant to be a sandbox for me, to explore how a fully/mostly
functional structure could look like! I'm really not married with it and we can
certainly do implement it in another way. It's just that I would like to write
down my ideas that are floating in my head! :)
"""

# TODO: imports
from typing import Tuple

import numpy.typing as npt


def delaymaker(num: int, d_min: float, d_max: float, *rng_stuff) -> Array:
    """Draws random delays for the delay array"""
    return delays


def draw_dendritic_weights(num, p, *rng_stuff) -> Array:
    """I need to look up how this works exacly... ;)"""
    return w_den


def draw_somatic_weights(num, p, *rng_stuff) -> [Array, Array]:
    """I need to look up how this works exacly... ;)
    But I think it is useful to also have a mask that stores what connections
    are actually present...?
    """
    return w_som, w_som_mask


# TODO: Buffer stuff...
class Buffer():

    def __init__(self, size, depth):
        """yet to be determined"""
        pass


def roll(buf: Buffer):
    return buf


def get(buffer, delays):
    return delayed_vars


class DataLoader():
    """Yet to be determined"""
    def __init__(self, sequence: Array, dt: floar, ):
        """We can pass one sequence, or many sequences that are shuffled,
        or regularly arranged, idk
        Maybe something like this already exists in pytorch or jax os sklearn?
        """
        pass

    def __call__(self, t: float) -> Array:
        """calculated what input we have at time t and returns it"""
        return pattern


# make me a dataclass
# and maybe move all the predefined parameters into a named tuple
class Network():
    """
    Struct like thing that holds all state variables (like u and v etc. ) and
    parameters (like g_l, E_l, etc.)

    """

    def __init__(self, config: dict|ConfigClass) -> None:

        # define constant parameters, these come from the paramfile:
        self.C_v = config.C_u|config["C_v"]
        self.C_u = etc...
        self.E_l =
        self.E_exc =
        self.E_inh =
        self.g_l =
        self.g_den =
        self.g_exc0 =
        self.g_inh0 =
        self.actfunc_a =
        self.actfunc_b =

        # (axonal) delays in units of seconds probably
        # (that means independet of dt)
        # requires some special thought on how to buffer...
        self.d_den = delaymaker(config.delay_params)
        self.d_som =

        # network size:
        self.num_vis = config.num_vis
        self.num_lat =
        self.num = self.num_vis + self.num_lat

        # weights:
        self.w_den = draw_dendritic_weights(self.num, *config.p_den, rng_stuff)
        self.w_som, self.w_som_mask = draw_somatic_weights(self.num, *config.p_som, *rng_stuff)

        # state variables
        self.u = Array(size=self.num, fill_value=self.E_l)
        self.v = Array(size=self.num, fill_value=self.E_l)
        self.r_bar = Array(size=self.num, fill_value=self.E_l)

        # rate buffer:
        self.r = Buffer()

        # weight optimizer:
        self.opt = SomeOptimizer(params) # like SGD or Adam or ...
        self.opt_state = self.opt.init()  # or however this works


# functions (all can be executed on GPU and jitted):
def _actfunc(u: Array, *params) -> Array:
    return r


def _calc_I_den(w_den: Array, r: Array, ):
    """eq. 8"""
    return I_den


def _calc_I_som(w_den, and_other_vars):
    """eq. 9 and 10 and 11"""
    return I_som


def _calc_input(w_den, inp, and_other_vars):
    """eq. 9 and 17 and 18"""
    return I_som


def _calc_u_dot(u, other_vars):
    """eq. 5"""
    return u_dot


def _calc_v_dot(v, other_vars):
    """eq. 4"""
    return v_dot


def _calc_v_bar_dot(r_bar, other_vars):
    """eq. 21"""
    return v_bar_dot


def _calc_w_den_dot(u, v_star, r_bar):
    """eq. 19"""
    return w_den_dot


def model_diffeq(t: float, net: Network, pattern):
    """basically calculates eq. 4, 5, 19 and 21"""

    # with approach, I think I broke a bit with the order we have in the pseudo-
    # code. But this can be fixed. It's more about the principle
    I_den = _calc_I_den(w_den, r)
    I_som = _calc_I_som(w_den, and_other_vars)
    I_som = _calc_input(w_den, inp, and_other_vars)
    u_dot = _calc_u_dot(u, other_vars)
    v_dot = _calc_v_dot(v, other_vars)
    v_bar_dot = _calc_v_bar_dot(r_bar, other_vars)
    w_den_dot = _calc_w_den_dot(u, v_star, r_bar)

    return u_dot, v_dot, r_bar_dot, W_den_dot


# the following isn't a pure function anymore because of the buffer
def update_step(t: float, [u, v, r_bar, W_den], net: Network, dataloader: DataLoader, ):
    net.u, net.v, net.r_bar, net.W_den = u, v, r_bar, W_den
    u_dot, v_dot, r_bar_dot, W_den_dot = model_diffeq(t, net, pattern)
    delta_w_den, net.opt_state = net.opt(W_den_dot, net.opt_state)
    net.r = roll(net.r, new_r)  # where does this new r come from? Honestly, this is tricky I have to admit! :/ I need to think about it

    return u_dot, v_dot, r_bar_dot, W_den_dot


# And then you can train the network with your favorite ODE-Solver:
res = some_odesolver(update_step, ts, args=(dataloader, ), solverargs)
