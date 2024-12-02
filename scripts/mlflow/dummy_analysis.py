#!/usr/bin/env python3

import matplotlib.pyplot as plt
import mlflow  # noqa
import numpy as np  # noqa
from mlflow.tracking import MlflowClient


def plot_results():
    client = MlflowClient()
    experiment = client.get_experiment_by_name("DummySim_Experiment")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        print("No runs found")
        return

    latest_run = runs[0]
    run_id = latest_run.info.run_id

    # Get the number of oscillators
    n_oscillators = int(latest_run.data.params["n_oscillators"])

    # Retrieve metrics
    metrics = client.get_metric_history(run_id, "position_0")
    steps = [m.step for m in metrics]

    # Plot positions
    plt.figure(figsize=(12, 6))
    for i in range(n_oscillators):
        positions = [
            m.value for m in client.get_metric_history(run_id, f"position_{i}")
        ]
        plt.plot(steps, positions, label=f"Oscillator {i}")

    plt.title("Oscillator Positions")
    plt.xlabel("Step")
    plt.ylabel("Position")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot velocities
    plt.figure(figsize=(12, 6))
    for i in range(n_oscillators):
        velocities = [
            m.value for m in client.get_metric_history(run_id, f"velocity_{i}")
        ]
        plt.plot(steps, velocities, label=f"Oscillator {i}")

    plt.title("Oscillator Velocities")
    plt.xlabel("Step")
    plt.ylabel("Velocity")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_results()
