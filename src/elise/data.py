"""Module for creating datasets and loading them into the model."""

# /usr/bin/env python3


# from abc import ABC, abstractmethod
from typing import Any, Callable, List

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
        self.dur = self.dt * self.__len__()
        self.shape = self.pattern.shape

    def __repr__(self) -> str:
        """Repr method."""
        return f"{self.__class__.__name__}(pattern={self._pattern}, dt={self.dt})"

    def __len__(self) -> int:
        """Return the length of the pattern."""
        return self.pattern.shape[0]

    def __getitem__(self, idx) -> Any:
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
    """Dataloader.

    maybe introduce pre- and online-transforms?

    """

    def __init__(
        self,
        pat: Pattern,
        pre_transforms: List[Callable] = [],
        online_transforms: List[Callable] = [],
    ) -> None:
        """Docstring."""
        self.pat = pat
        self.dur = pat.dur
        self.dt = pat.dt

        self.online_transforms = online_transforms

        # apply pre-transforms directly once
        for transform in pre_transforms:
            self.pat.pattern = transform(self.pat.pattern)
            if self.pat.pattern.shape != self.pat.shape:
                raise ValueError(
                    "The pre_transform {transform} may not change the shape of the pattern!"  # noqa
                )

    def _time_to_idx(self, t: float) -> int:
        return int((t % self.dur) / self.dt)

    def _apply_online_transforms(self, pat_1d):
        for transform in self.online_transforms:
            pat_1d = transform(pat_1d)
        return pat_1d

    def __call__(self, t: float, offset: float = 1e-6):
        """When the dataloader is called, the correct pattern at time t is returned.

        example:
        for t in np.arange(0, 100, 0.1):
            pat = dataloader(t)
            my_simulation.step(t, u_inp=pat,...)
        """
        idx = self._time_to_idx(t + offset)
        pat_t = self.pat[idx]

        pat_t = self._apply_online_transforms(pat_t)

        return pat_t

    def iter(self, t_start, t_stop, dt):
        """Use dataloader as an iterator/iterable.

        example:
        for t, pat in dataloader.iter(t_start, t_stop, dt):
            my_simulation.step(t, u_inp=pat,...)
        """
        return self.Iterator(...)
