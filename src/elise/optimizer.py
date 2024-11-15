#!/usr/bin/env python3
from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, eta):
        self.eta = eta

    @abstractmethod
    def update(self, weights, step):
        raise NotImplementedError("Subclasses must implement this method")


class SimpleUpdater(Optimizer):
    def update(self, weights, step):
        return weights + (step * self.eta)
