import numpy as np
from dummy_sim import DummySim
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver  # noqa

ex = Experiment("oscillator")
ex.observers.append(FileStorageObserver("oscillator_res"))
# print("BEFORE MongoDB")
# ex.observers.append(MongoObserver(url="mongodb://localhost:27017", db_name="sacred"))
# print("AFTER MongoDB")


@ex.config
def config():
    n_osci = 5  # noqa
    total_time = 5  # noqa
    dt = 0.1  # noqa
    seed = 42  # noqa


@ex.automain
def run(_run, n_osci, total_time, dt, seed):
    sim = DummySim(n_oscillators=n_osci, dt=dt, seed=seed)
    print("SIMULATION STARTS")
    res = sim.simulate(total_time)
    print("SIMULATION DONE")

    # Save results as .npz file
    np.savez(
        "simulation_results.npz",
        time=res["time"],
        positions=res["positions"],
        velocities=res["velocities"],
    )

    # Add the .npz file as an artifact
    _run.add_artifact("simulation_results.npz")

    # Log the result
    for i in range(len(res["time"])):
        _run.log_scalar("time", res["time"][i], i)
        for j in range(n_osci):
            _run.log_scalar(f"position_{j}", res["positions"][i, j], i)
            _run.log_scalar(f"velocity_{j}", res["velocities"][i, j], i)


if __name__ == "__main__":
    ex.run()
