#!/usr/bin/env python3

import time

import cupy as cp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch


def numpy_init(n):
    start_time = time.time()
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


def jax_cpu_init(n):
    start_time = time.time()
    jax.config.update("jax_platform_name", "cpu")
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
    # Warm-up call to compile the function
    _ = jax_cpu_compute_jit(a, M, b).block_until_ready()

    start_time = time.time()
    result = jax_cpu_compute_jit(a, M, b).block_until_ready()
    end_time = time.time()
    return end_time - start_time


def jax_gpu_init(n):
    start_time = time.time()
    jax.config.update("jax_platform_name", "gpu")
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


def pytorch_cpu_init(n):
    start_time = time.time()
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


def pytorch_gpu_init(n):
    start_time = time.time()
    a = torch.rand(n, device="cuda")
    M = torch.rand(n, n, device="cuda")
    b = torch.rand(n, device="cuda")
    end_time = time.time()
    return (a, M, b), end_time - start_time


def pytorch_gpu_compute(a, M, b):
    start_time = time.time()
    result = torch.matmul(a, M) + b
    end_time = time.time()
    return end_time - start_time


def cupy_init(n):
    start_time = time.time()
    a = cp.random.rand(n)
    M = cp.random.rand(n, n)
    b = cp.random.rand(n)
    end_time = time.time()
    return (a, M, b), end_time - start_time


def cupy_compute(a, M, b):
    start_time = time.time()
    result = cp.dot(a, M) + b
    end_time = time.time()
    return end_time - start_time


def test_function(n, library):
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
        data, init_time = init_func(n)
        compute_time = compute_func(*data)
        return init_time, compute_time
    except KeyError:
        raise ValueError(f"Unsupported library: {library}")


def plot_results(results, n_values):
    libraries = list(results.keys())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))

    for lib in libraries:
        init_times = results[lib]["init"]
        compute_times = results[lib]["compute"]

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

    ax1.set_title("Initialization Time")
    ax2.set_title("Computation Time")

    plt.tight_layout()
    plt.savefig("matrix_operation_benchmark.png")
    plt.show()


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
    results = {lib: {"init": [], "compute": []} for lib in libraries}

    for n in n_values:
        print(f"Testing for n = {n}")
        for lib in libraries:
            try:
                init_time, compute_time = test_function(n, lib)
                results[lib]["init"].append(init_time)
                results[lib]["compute"].append(compute_time)
                print(f"{lib}: Init: {init_time:.6f} s, Compute: {compute_time:.6f} s")
            except Exception as e:
                print(f"Error with {lib}: {str(e)}")
                results[lib]["init"].append(0)
                results[lib]["compute"].append(0)
        print()

    return results


if __name__ == "__main__":
    n_values = np.logspace(4, 14, base=2.0, num=11, dtype=int)
    results = run_tests(n_values)

    print("Summary of results:")
    for lib in results:
        print(f"{lib}:")
        print(f"  Init times: {results[lib]['init']}")
        print(f"  Compute times: {results[lib]['compute']}")

    plot_results(results, n_values)
