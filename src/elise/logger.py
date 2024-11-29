#!/usr/bin/env python3


class Logger:
    def __init__(self, log_file: str):
        self.log_file = log_file

    def log(self, message: str):
        with open(self.log_file, "a") as f:
            f.write(message + "\n")


# taken directly from Laura
class Tracker:
    """
    Tracks/records changes in 'target' array. Records 'length'*'compress_len'
    samples, compressed (averaged) into 'length' samples. The result is stored
    in 'data'. Note that the first value in the 'data' is already the average
    of multiple values of the target array. If 'compress_len' is not 1 the
    initial value of 'target' is therefore not equal to the first entry in 'data'.
    After recording call finalize to also add the remaining data in buffer to
    'data' (finish the last compression).
    """

    def __init__(self, length, target, compress_len):
        self.target = target
        self.data = np.zeros(tuple([length]) + target.shape, dtype=np.float32)
        self.index = 0
        self.buffer = np.zeros(target.shape)
        self.din = compress_len

    def record(self):
        self.buffer += self.target
        if (self.index + 1) % self.din == 0:
            self.data[int(self.index / self.din), :] = self.buffer / self.din
            self.buffer.fill(0)
        self.index += 1

    def finalize(self):
        """fill last data point with average of remaining target data in buffer."""
        n_buffer = self.index % self.din
        if n_buffer > 0:
            self.data[int(self.index / self.din), :] = self.buffer / n_buffer


if __name__ == "__main__":
    n = 10
    t_max = 1e3
    dt = 0.1

    arr = np.zeros(n)
    tracker = Tracker

    def some_testfunc(t, arr, omega, phi):
        return np.sin(ts / omega - phi)

    omega = 100.0
    phis = np.linspace(0, 0.6, n)

    t = 0.0
    while t < t_max:
        arr[:] = some_testfunc(t, arr, omega, phis)
        t += dt
