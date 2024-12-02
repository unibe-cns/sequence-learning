import numpy as np


class DummySim:
    def __init__(self, n_oscillators, dt=0.01, seed=None):
        """
        Initialize the simulation with n harmonic oscillators.

        :param n_oscillators: Number of oscillators to simulate
        :param dt: Time step for the simulation (default: 0.01)
        """
        self.n = n_oscillators
        self.dt = dt

        self.rng = np.random.default_rng(seed)

        # Initialize random frequencies and phase shifts for each oscillator
        self.frequencies = np.random.uniform(0.5, 2.0, self.n)
        self.phase_shifts = np.random.uniform(0, 2 * np.pi, self.n)

        # Initialize positions and velocities
        self.positions = self.rng.uniform(-1, 1, self.n)
        self.velocities = self.rng.normal(-0.1, 0.1)

        self.time = 0

    def step(self):
        """
        Perform a single time step using forward Euler integration.
        """
        # Update velocities
        accelerations = -self.frequencies**2 * self.positions
        self.velocities += accelerations * self.dt

        # Update positions
        self.positions += self.velocities * self.dt

        # Update time
        self.time += self.dt

    def simulate(self, total_time):
        """
        Run the simulation for a specified total time.

        :param total_time: Total simulation time
        :return: Dictionary containing time series of positions and time
        """
        n_steps = int(total_time / self.dt)
        positions = np.zeros((n_steps, self.n))
        velocities = np.zeros((n_steps, self.n))
        time_points = np.zeros(n_steps)

        for i in range(n_steps):
            print(f"STEP {i}")
            positions[i] = self.positions
            velocities[i] = self.velocities
            time_points[i] = self.time
            self.step()

        return {"time": time_points, "positions": positions, "velocities": velocities}


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sim = DummySim(10)
    res = sim.simulate(10.0)
    plt.plot(res["time"], res["positions"])
    plt.show()
