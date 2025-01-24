#!/usr/bin/env python3

from collections import defaultdict

import dill
import numpy as np

from .model import Network


class Tracker:
    def __init__(
        self,
        network: Network,
        variables: list[tuple[str, str]],
        track_step: int,
        epoch_duration: int,
    ):
        if not all(
            isinstance(var, (list, tuple)) and len(var) == 2 for var in variables
        ):
            raise ValueError(
                "Variables must be list of lists with two elements (name, view)."
            )
        self.network = network
        self.variables = variables
        self.track_step = track_step
        self.track_counter = 0
        self.track_dict = defaultdict(list)
        self.epoch_duration = epoch_duration
        self.effective_epoch_duration = epoch_duration // track_step

    def _get_variable_value(self, name: str, view: str):
        try:
            return self.network.get_val(name, view)
        except Exception as e:
            print(f"Error retrieving {name} ({view}): {str(e)}")
            return None

    def track(self, time):
        self.track_dict["time"].append(time)
        self.track_counter += 1
        if self.track_counter % self.track_step == 0:
            for name, view in self.variables:
                key = f"{name}_{view}"
                val = self._get_variable_value(name, view)
                if val is not None:  # Only store valid results
                    self.track_dict[key].append(val)

    def save(self, path):
        with open(path, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return dill.load(f)

    def __getitem__(self, item):
        return np.array(self.track_dict[item])

    def get_epochs(self, item, epoch_nums):
        if isinstance(epoch_nums, (int, np.integer)):
            epoch_nums = [epoch_nums]

        epochs = []
        for epoch_num in epoch_nums:
            epoch_slice = np.s_[
                epoch_num
                * self.effective_epoch_duration : (epoch_num + 1)
                * self.effective_epoch_duration
            ]
            epochs.append(np.array(self.track_dict[item])[epoch_slice])

        return np.concatenate(epochs)

        def __repr__(self):
            return (
                f"Tracker({list(self.track_dict.keys())}, "
                f"length={self.track_counter}, "
                f"track_step={self.track_step})"
            )
