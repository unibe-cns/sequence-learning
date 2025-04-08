#!/usr/bin/env python3

"""Run a whole script, for instance as a smoke test."""

import subprocess
from pathlib import Path

import pytest

path = Path(__file__).parent.resolve()


@pytest.mark.parametrize(
    "script_path", [path / "experiment_smoketest.py"]
)  # Replace with your script path
def test_script_execution(script_path):
    try:
        # Run the script using subprocess
        result = subprocess.run(
            ["python", script_path],
            check=True,  # Raises CalledProcessError if the script fails
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Decodes output to string
        )
        # Assert that the script ran successfully
        assert (
            result.returncode == 0
        ), f"Script failed with return code {result.returncode}"
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Script execution failed! Error:\n{e.stderr}")  # noqa
    except FileNotFoundError:
        pytest.fail("The specified script file was not found.")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred: {e}")
