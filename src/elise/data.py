"""Module for creating datasets and loading them into the model."""

# /usr/bin/env python3


from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional

import numpy as np
import numpy.typing as npt


class BasePattern(ABC):
    """
    Base Class for all pattern types.

    More specific patterns should inherit from this.
    Defines the interface for all other Pattern classes.

    :ivar _pattern: The original input pattern array
    :vartype _pattern: npt.NDArray
    :ivar pattern: The converted pattern array
    :vartype pattern: npt.NDArray
    :ivar dt: Time step
    :vartype dt: float
    :ivar dur: Total duration of the pattern
    :vartype dur: float
    :ivar shape: Shape of the pattern array
    :vartype shape: tuple
    """

    def __init__(
        self,
        pattern: npt.NDArray,
        dt: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> None:
        """
        Initialize the Pattern object.

        :param pattern: The input pattern array.
        :type pattern: npt.NDArray
        :param dt: Time step for the pattern (mutually exclusive with 'duration').
        :type dt: Optional[float]
        :param duration: Total duration of the pattern (mutually exclusive with 'dt').
        :type duration: Optional[float]
        :raises ValueError: If neither or both 'dt' and 'duration' are provided.

        Calculates either duration or time step based on provided arguments.

        Attributes:
            _pattern (npt.NDArray): Original input pattern array.
            pattern (npt.NDArray): Converted pattern array.
            dt (float): Time step if specified.
            duration (float): Total duration if specified.
            shape (tuple): Shape of the converted pattern array.
        """
        self._pattern = pattern
        self.pattern = self._convert(pattern)
        if (dt is None and duration is None) or (
            dt is not None and duration is not None
        ):
            raise ValueError("Exactly one of 'dt' or 'duration' must be provided")
        if type(dt) is float:
            self.dt = dt
            self.duration = self.dt * self.__len__()
        if type(duration) is float:
            self.duration = duration
            self.dt = self.duration / self.__len__()
        self.shape = self.pattern.shape

    @abstractmethod
    def _convert(self, pattern: npt.NDArray) -> npt.NDArray:
        """
        Convert the input pattern.

        Basically one of the core functionalities of the Pattern-Classes.
        Every Pattern class needs to implement this.

        :param pattern: The input pattern array
        :type pattern: npt.NDArray
        :return: The converted pattern
        :rtype: npt.NDArray
        """
        pass

    def transform(self, transformation: Callable[[npt.NDArray], npt.NDArray]) -> None:
        """Apply a transformation to the whole pattern.

        At the moment, the transformation takes only a pattern, no other arguments.
        """
        self.pattern = transformation(self.pattern)
        if self.shape != self.pattern.shape:
            raise ValueError(
                "The transformation {transform} may not change the shape of the pattern!"  # noqa
            )

    def __repr__(self) -> str:
        """
        Return a string representation of the Pattern object.

        :return: String representation
        :rtype: str
        """
        return f"{self.__class__.__name__}(pattern={self._pattern}, dt={self.dt})"

    def __str__(self) -> str:
        """
        Return a string representation.
        """
        return "\n".join(["pattern=", str(self.pattern), f"dt={self.dt}"])

    def __len__(self) -> int:
        """
        Return the length of the pattern.

        :return: Length of the pattern
        :rtype: int
        """
        return self.pattern.shape[0]

    def __getitem__(self, idx) -> Any:
        """
        Get item at the specified index.

        :param idx: Index
        :type idx: int
        :return: Item at the specified index
        :rtype: Any
        """
        return self.pattern[idx]


class Pattern(BasePattern):
    """Simplest Pattern class.

    No conversion happens.
    """

    def _convert(self, pattern: npt.NDArray):
        """No conversion."""
        return pattern


class Pattern2:
    """TODO: rewrite Pattern with properties:

    - pattern should be a property
    - and t_max also which is changed whenever pattern is changed
    - compare also in terms of speed
    """

    pass


class OneHotPattern(BasePattern):
    """
    Turn a sequential pattern into a one-hot encoded pattern.

    This class extends the base Pattern class to create one-hot encoded patterns.

    :ivar _width: The width of the one-hot encoded pattern
    :vartype _width: int
    """

    def __init__(self, pattern: npt.NDArray, dt: float, width: int) -> None:
        """
        Initialize the OneHotPattern object.

        :param pattern: The input sequential pattern array
        :type pattern: npt.NDArray
        :param dt: Time step
        :type dt: float
        :param width: Width of the one-hot encoded pattern
        :type width: int
        """
        if len(pattern.shape) != 1:
            raise ValueError("pattern must be one dimensional")
        if np.max(pattern) > width - 1:
            raise ValueError(
                "width must be greater then or equal to the maximum value in pattern"
            )
        self._width = width
        super().__init__(pattern, dt)

    def _convert(self, pattern):
        res = np.zeros((len(pattern), self._width))
        for i, j in enumerate(pattern):
            res[i, j] = 1.0
        return res


class MultiHotPattern(BasePattern):
    """
    Same as OneHot, but more then one can be active at a time.

    Encode it as:
    [2, 4, [0, 3], [0, 3], -1]

    If an entry is -1, there's no, the whole vector is zero (= a pause in the pattern)
    """

    def __init__(
        self, pattern: List[int | List[int]], duration: float, width: int
    ) -> None:
        """DOCSTRING."""
        # max in list:
        max_val = -1000000
        for i in pattern:
            if isinstance(i, int) and i > max_val:
                max_val = i
            elif isinstance(i, list):
                for j in i:
                    if j > max_val:
                        max_val = j
        if max_val > width - 1:
            raise ValueError(
                "width must be greater then or equal to the maximum value in pattern"
            )
        self._width = width
        super().__init__(pattern, duration=duration)

    def _convert(self, pattern):
        res = np.zeros((len(pattern), self._width))
        for i, pat in enumerate(pattern):
            if isinstance(pat, int) and pat == -1:
                continue
            else:
                res[i, pat] = 1.0
        return res


class Dataloader:
    """
    Dataloader for pattern data.

    Has a call-method for passing the correct pattern at time t.
    And an iter-method that acts as an iterator.
    Handles pre-transforms and online-transforms for pattern data.

    :ivar pat: The pattern object
    :vartype pat: Pattern
    :ivar dur: Duration of the pattern
    :vartype dur: float
    :ivar dt: Time step of the pattern
    :vartype dt: float
    :ivar online_transforms: List of online transforms to be applied
    :vartype online_transforms: List[Callable]
    """

    def __init__(
        self,
        pattern: Pattern,
        pre_transforms: List[Callable] = [],
        online_transforms: List[Callable] = [],
    ) -> None:
        """
        Initialize the Dataloader object.

        :param pattern: The pattern object
        :type pattern: Pattern
        :param pre_transforms: List of pre-transforms to be applied once, defaults to []
        :type pre_transforms: List[Callable], optional
        :param online_transforms: List of online transforms to be applied on each call, defaults to []  # noqa
        :type online_transforms: List[Callable], optional
        :raises ValueError: If a pre-transform changes the shape of the pattern
        """
        self.pattern = pattern
        self.duration = self.pattern.duration
        self.dt = self.pattern.dt

        self.online_transforms = online_transforms

        # apply pre-transforms directly once
        for transform in pre_transforms:
            self.pattern.transform(transform)

    def _time_to_idx(self, t: float) -> int:
        return int((t % self.duration) / self.dt)

    def _apply_online_transforms(self, pattern_1d):
        for transform in self.online_transforms:
            pattern_1d = transform(pattern_1d)
        return pattern_1d

    def __call__(self, t: float, offset: float = 1e-6):
        """
        Return the correct pattern at time t.

        The offset is a small value added to t to make sure that the pattern is
        read "in the middle". This is to prevent that floating point glitches read
        from the wrong pattern.

        :param t: Time
        :type t: float
        :param offset: Time offset, defaults to 1e-6
        :type offset: float, optional
        :return: Pattern at time t
        :rtype: npt.NDArray

        Example:
            for t in np.arange(0, 100, 0.1):
                pattern = dataloader(t)
                my_simulation.step(t, u_inp=pattern,...)
        """
        idx = self._time_to_idx(t + offset)
        pattern_t = self.pattern[idx]

        pattern_t = self._apply_online_transforms(pattern_t)

        return pattern_t

    def iter(self, t_start, t_stop, dt):
        """
        Use dataloader as an iterator/iterable.

        :param t_start: Start time
        :type t_start: float
        :param t_stop: Stop time
        :type t_stop: float
        :param dt: Time step
        :type dt: float
        :yield: Tuple of time and pattern
        :rtype: Tuple[float, npt.NDArray]

        Example:
            for t, pattern in dataloader.iter(t_start, t_stop, dt):
                my_simulation.step(t, u_inp=pattern,...)
        """
        t = t_start
        while t < t_stop:
            yield t, self.__call__(t, offset=dt * 0.01)
            t += dt

    def get_full_pattern(self, dt):
        """Return the full pattern."""
        """
        :param dt: Simulation time step
        :type dt: float
        :return: Full pattern
        :rtype: npt.NDArray
        """

        full_pattern = []
        for _, pattern in self.iter(0, self.duration, dt):
            full_pattern.append(pattern)

        return np.array(full_pattern)
