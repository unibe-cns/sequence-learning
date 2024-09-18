#!/usr/bin/env python3

from abc import ABC, abstractmethod


class Buffer(ABC):
    def __init__(self, buffer_size: int):
        self.size = None
        self.depth = None
        self.buffer = None

    @abstractmethod
    def get_buffer(self):
        pass

    @abstractmethod
    def roll(self, update):
        pass


# Your amazing buffer goes here
