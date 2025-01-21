#!/usr/bin/env python3

import mlflow
import numpy as np
from dummy_sim import DummySim

# start the mlfow server with:
# mlflow server --backend-store-uri ./mlruns


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DummySim_Experiment")


def run_experiment(n_oscillators=5, total_time=5, dt=0.1, seed=42):
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_oscillators", n_oscillators)
        mlflow.log_param("total_time", total_time)
        mlflow.log_param("dt", dt)
        mlflow.log_param("seed", seed)

        # Run simulation
        sim = DummySim(n_oscillators=n_oscillators, dt=dt, seed=seed)
        results = sim.simulate(total_time)

        # Log metrics
        for step in range(len(results["time"])):
            for i in range(n_oscillators):
                mlflow.log_metric(
                    f"position_{i}", results["positions"][step, i], step=step
                )
                mlflow.log_metric(
                    f"velocity_{i}", results["velocities"][step, i], step=step
                )

        # Log the final arrays as artifacts
        np.save("positions", results["positions"])
        np.save("velocities", results["velocities"])
        mlflow.log_artifact("positions.npy")
        mlflow.log_artifact("velocities.npy")


if __name__ == "__main__":
    run_experiment()
