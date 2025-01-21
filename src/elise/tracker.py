#!/usr/bin/env python3

from collections import defaultdict

from .model import Network


class Tracker:
    def __init__(self, network: Network, variables: list, track_step: int = 1):
        self.network = network
        self.variables = variables
        self.track_step = track_step
        self.track_counter = 0
        self.track_dict = defaultdict(list)

    def track(self):
        self.track_counter += 1
        if self.track_counter % self.track_step == 0:
            try:
                for name, view in self.variables:
                    key = f"{name}_{view}"
                    val = self.network.get_val(name, view)
                    self.track_dict[key].append(val)

            except Exception as e:
                print(f"Error during tracking: {str(e)}")
        else:
            pass
