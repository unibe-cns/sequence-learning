"""Module for creating datasets and loading them into the model."""
# /usr/bin/env python3


# from abc import ABC, abstractmethod

# import numpy as np
import numpy.typing as npt


class Pattern:
    """Base Class for all pattern types.

    More specific pattern should inherit from this.
    Defines the interface for all other Pattern classes.
    """

    def __init__(self, pattern: npt.NDArray, dt: float) -> None:
        """Docstring."""
        self._pattern = pattern
        self.pattern = self._pattern  # do nothing here!
        self.dt = dt
        self.t_max = self.dt * self.__len__()
        self.shape = self.pattern.shape

    def __repr__(self) -> str:
        """Repr method."""
        return f"{self.__class__.__name__}(pattern={self._pattern}, dt={self.dt})"

    def __len__(self) -> int:
        """Return the length of the pattern."""
        return self.pattern.shape[0]

    def __getitem__(self, idx):
        """Get item."""
        return self.pattern[idx]


class Pattern2:
    """TODO: rewrite Pattern with properties:

    - pattern should be a property
    - and t_max also which is changed whenever pattern is changed
    - compare also in terms of speed
    """

    pass


class SequentialPattern(Pattern):
    """Turn a sequential pattern into a propper pattern."""

    pass


class Dataloader:
    def __init__(self):
        pass
