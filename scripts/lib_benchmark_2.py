#!/usr/bin/env python3

import time

import cupy as cp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch


def create_deterministic_data(n):
    # Create a deterministic pattern
    return (
        np.arange(n) / n,
        np.arange(n * n).reshape(n, n) / (n * n),
        np.arange(n)[::-1] / n,
    )


# NumPy functions
def numpy_init(n, deterministic=False):
    start_time = time.time()
    if deterministic:
        a, M, b = create_deterministic_data(n)
    else:
        a = np.random.rand(n)
        M = np.random.rand(n, n)
        b = np.random.rand(n)
    end_time = time.time()
    return (a, M, b), end_time - start_time


def numpy_compute(a, M, b):
    start_time = time.time()
    result = np.dot(a, M) + b
    end_time = time.time()
    return end_time - start_time


# JAX CPU functions
def jax_cpu_init(n, deterministic=False):
    jax.config.update("jax_platform_name", "cpu")
    start_time = time.time()
    if deterministic:
        a, M, b = [jnp.array(x) for x in create_deterministic_data(n)]
    else:
        key1, key2, key3 = jax.random.split(jax.random.PRNGKey(0), 3)
        a = jax.random.uniform(key1, (n,))
        M = jax.random.uniform(key2, (n, n))
        b = jax.random.uniform(key3, (n,))
    end_time = time.time()
    return (a, M, b), end_time - start_time


def jax_cpu_compute(a, M, b):
    start_time = time.time()
    result = jnp.dot(a, M) + b
    end_time = time.time()
    return end_time - start_time


@jax.jit
def jax_cpu_compute_jit(a, M, b):
    return jnp.dot(a, M) + b


def jax_cpu_compute_with_jit(a, M, b):
    _ = jax_cpu_compute_jit(a, M, b).block_until_ready()
    start_time = time.time()
    result = jax_cpu_compute_jit(a, M, b).block_until_ready()
    end_time = time.time()
    return end_time - start_time


# JAX GPU functions (similar to CPU, but with GPU config)
def jax_gpu_init(n, deterministic=False):
    jax.config.update("jax_platform_name", "gpu")
    start_time = time.time()
    if deterministic:
        a, M, b = [jnp.array(x) for x in create_deterministic_data(n)]
    else:
        key1, key2, key3 = jax.random.split(jax.random.PRNGKey(0), 3)
        a = jax.random.uniform(key1, (n,))
        M = jax.random.uniform(key2, (n, n))
        b = jax.random.uniform(key3, (n,))
    end_time = time.time()
    return (a, M, b), end_time - start_time


def jax_gpu_compute(a, M, b):
    start_time = time.time()
    result = jnp.dot(a, M) + b
    end_time = time.time()
    return end_time - start_time


@jax.jit
def jax_gpu_compute_jit(a, M, b):
    return jnp.dot(a, M) + b


def jax_gpu_compute_with_jit(a, M, b):
    # Warm-up call to compile the function
    _ = jax_gpu_compute_jit(a, M, b).block_until_ready()

    start_time = time.time()
    result = jax_gpu_compute_jit(a, M, b).block_until_ready()
    end_time = time.time()
    return end_time - start_time


# PyTorch CPU functions
def pytorch_cpu_init(n, deterministic=False):
    start_time = time.time()
    if deterministic:
        a, M, b = [
            torch.tensor(x, dtype=torch.float32) for x in create_deterministic_data(n)
        ]
    else:
        a = torch.rand(n)
        M = torch.rand(n, n)
        b = torch.rand(n)
    end_time = time.time()
    return (a, M, b), end_time - start_time


def pytorch_cpu_compute(a, M, b):
    start_time = time.time()
    result = torch.matmul(a, M) + b
    end_time = time.time()
    return end_time - start_time


# PyTorch GPU functions
def pytorch_gpu_init(n, deterministic=False):
    start_time = time.time()
    if deterministic:
        a, M, b = [
            torch.tensor(x, dtype=torch.float32, device="cuda")
            for x in create_deterministic_data(n)
        ]
    else:
        a = torch.rand(n, device="cuda")
        M = torch.rand(n, n, device="cuda")
        b = torch.rand(n, device="cuda")
    end_time = time.time()
    return (a, M, b), end_time - start_time


def pytorch_gpu_compute(a, M, b):
    start_time = time.time()
    result = torch.matmul(a, M) + b
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


# CuPy functions
def cupy_init(n, deterministic=False):
    start_time = time.time()
    if deterministic:
        a, M, b = [cp.array(x) for x in create_deterministic_data(n)]
    else:
        a = cp.random.rand(n)
        M = cp.random.rand(n, n)
        b = cp.random.rand(n)
    cp.cuda.Stream.null.synchronize()
    end_time = time.time()
    return (a, M, b), end_time - start_time


def cupy_compute(a, M, b):
    start_time = time.time()
    result = cp.dot(a, M) + b
    cp.cuda.Stream.null.synchronize()
    end_time = time.time()
    return end_time - start_time


def test_function(n, library, deterministic=False):
    operations = {
        "numpy": (numpy_init, numpy_compute),
        "jax_cpu": (jax_cpu_init, jax_cpu_compute),
        "jax_cpu_jit": (jax_cpu_init, jax_cpu_compute_with_jit),
        "jax_gpu": (jax_gpu_init, jax_gpu_compute),
        "jax_gpu_jit": (jax_gpu_init, jax_gpu_compute_with_jit),
        "pytorch_cpu": (pytorch_cpu_init, pytorch_cpu_compute),
        "pytorch_gpu": (pytorch_gpu_init, pytorch_gpu_compute),
        "cupy": (cupy_init, cupy_compute),
    }

    try:
        init_func, compute_func = operations[library]
        data, init_time = init_func(n, deterministic)
        compute_time = compute_func(*data)
        return init_time, compute_time
    except KeyError:
        raise ValueError(f"Unsupported library: {library}")


def plot_results(results, n_values):
    libraries = list(results.keys())

    for init_type in ["random", "deterministic"]:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))

        for lib in libraries:
            init_times = results[lib][f"init_{init_type}"]
            compute_times = results[lib][f"compute_{init_type}"]

            ax1.plot(n_values, init_times, marker="o", label=lib)
            ax2.plot(n_values, compute_times, marker="o", label=lib)

        for ax in (ax1, ax2):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Problem Size (n)")
            ax.set_ylabel("Time (seconds)")
            ax.legend()
            ax.grid(True, which="both", ls="-", alpha=0.2)
            ax.set_xticks(n_values)
            ax.set_xticklabels(n_values)

        ax1.set_title(f"Initialization Time ({init_type.capitalize()})")
        ax2.set_title(f"Computation Time ({init_type.capitalize()})")

        plt.tight_layout()
        plt.savefig(f"matrix_operation_benchmark_{init_type}.png")
        plt.close()


def run_tests(n_values):
    libraries = [
        "numpy",
        "jax_cpu",
        "jax_cpu_jit",
        "jax_gpu",
        "jax_gpu_jit",
        "pytorch_cpu",
        "pytorch_gpu",
        "cupy",
    ]
    results = {
        lib: {
            "init_random": [],
            "compute_random": [],
            "init_deterministic": [],
            "compute_deterministic": [],
        }
        for lib in libraries
    }

    for n in n_values:
        print(f"Testing for n = {n}")
        for lib in libraries:
            try:
                # Random initialization
                init_time, compute_time = test_function(n, lib, deterministic=False)
                results[lib]["init_random"].append(init_time)
                results[lib]["compute_random"].append(compute_time)
                print(
                    f"{lib} (Random): Init: {init_time:.6f} s, Compute: {compute_time:.6f} s"
                )

                # Deterministic initialization
                init_time, compute_time = test_function(n, lib, deterministic=True)
                results[lib]["init_deterministic"].append(init_time)
                results[lib]["compute_deterministic"].append(compute_time)
                print(
                    f"{lib} (Deterministic): Init: {init_time:.6f} s, Compute: {compute_time:.6f} s"
                )
            except Exception as e:
                print(f"Error with {lib}: {str(e)}")
                results[lib]["init_random"].append(0)
                results[lib]["compute_random"].append(0)
                results[lib]["init_deterministic"].append(0)
                results[lib]["compute_deterministic"].append(0)
        print()

    return results


if __name__ == "__main__":
    n_values = np.logspace(4, 14, base=2.0, num=11, dtype=int)
    results = run_tests(n_values)

    print("Summary of results:")
    for lib in results:
        print(f"{lib}:")
        print(f"  Random Init times: {results[lib]['init_random']}")
        print(f"  Random Compute times: {results[lib]['compute_random']}")
        print(f"  Deterministic Init times: {results[lib]['init_deterministic']}")
        print(f"  Deterministic Compute times: {results[lib]['compute_deterministic']}")

    plot_results(results, n_values)
