#!/usr/bin/env python3

import json
import os

import matplotlib.pyplot as plt
import numpy as np


def get_latest_run_dir(base_dir):
    run_dirs = [d for d in os.scandir(base_dir) if d.is_dir()]
    return max(run_dirs, key=lambda x: x.stat().st_mtime).path


def plot_from_file():
    base_dir = "oscillator_res"
    latest_run_dir = get_latest_run_dir(base_dir)

    with open(os.path.join(latest_run_dir, "metrics.json"), "r") as f:
        metrics = json.load(f)

    time = np.array(metrics["time"]["values"])
    n_oscillators = len([key for key in metrics.keys() if key.startswith("position_")])

    plt.figure(figsize=(12, 8))
    for i in range(n_oscillators):
        positions = np.array(metrics[f"position_{i}"]["values"])
        plt.plot(time, positions, label=f"Oscillator {i}")

    plt.title("Harmonic Oscillators Simulation")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_from_file()
