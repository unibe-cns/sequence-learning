#!/usr/bin/env python3

"""
Functions for evaluating the model performance.
"""
from typing import Callable

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import pearsonr


def window_slider(
    data: npt.NDArray, target: npt.NDArray, func: Callable[[npt.NDArray], float]
):
    """
    Apply a sliding window operation on data using a given function.

    :param data: Input data array
    :type data: npt.NDArray
    :param target: Target array defining the window shape
    :type target: npt.NDArray
    :param func: Function to apply on each window
    :type func: Callable[[npt.NDArray, npt.NDArray], float]
    :return: Array of results from applying func on each window
    :rtype: npt.NDArray
    """
    windows = sliding_window_view(data, target.shape).squeeze()
    return np.array([func(chunk, target) for chunk in windows])


def pearson_coef_(data: npt.NDArray, target: npt.NDArray):
    """
    Calculate the mean Pearson correlation coefficient between data and target.

    THIS VERSION REQUIRES SCIPY VERSION 1.14!

    :param data: Input data array
    :type data: npt.NDArray
    :param target: Target array
    :type target: npt.NDArray
    :return: Mean Pearson correlation coefficient
    :rtype: float
    """
    p_coefs = pearsonr(data, target, axis=0).statistic
    return np.mean(p_coefs)


def pearson_coef(data: npt.NDArray, target: npt.NDArray):
    """
    Calculate the mean Pearson correlation coefficient between data and target.

    :param data: Input data array
    :type data: npt.NDArray
    :param target: Target array
    :type target: npt.NDArray
    :return: Mean Pearson correlation coefficient
    :rtype: float
    """
    p_coefs = []
    for d, t in zip(
        np.nditer(data, order="F", flags=["external_loop"]),
        np.nditer(target, order="F", flags=["external_loop"]),
    ):
        p_coefs.append(pearsonr(d, t).statistic)
    return np.mean(p_coefs)


def mse(data: npt.NDArray, target: npt.NDArray):
    """
    Calculate the Mean Squared Error between data and target.

    :param data: Input data array
    :type data: npt.NDArray
    :param target: Target array
    :type target: npt.NDArray
    :return: Mean Squared Error
    :rtype: float
    """
    return np.mean((data - target) ** 2)
