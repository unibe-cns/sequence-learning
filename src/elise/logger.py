#!/usr/bin/env python3


class Logger:
    def __init__(self, log_file: str):
        self.log_file = log_file

    def log(self, message: str):
        with open(self.log_file, "a") as f:
            f.write(message + "\n")
