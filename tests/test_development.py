#!/usr/bin/env python3

import unittest

import numpy as np

from seqlearn.main import (
    SomaticWeights,  # Replace 'your_module' with the actual module name
)


class TestSomaticWeights(unittest.TestCase):
    def setUp(self):
        # Create a mock config for testing
        self.config = {
            "weight_params": {
                "latent_neurons": 100,
                "output_neurons": 10,
                "p": 0.5,
                "q": 0.3,
                "p0": 0.1,
            }
        }
        self.somatic_weights = SomaticWeights(self.config)

    def test_weight_matrix_shape(self):
        weight_matrix, _, _ = self.somatic_weights.create_weight_matrix(
            self.config["weight_params"]["p"],
            self.config["weight_params"]["q"],
            self.config["weight_params"]["p0"],
            self.config["weight_params"]["latent_neurons"],
            self.config["weight_params"]["output_neurons"],
        )
        self.assertEqual(weight_matrix.shape, (100, 100))

    def test_weight_matrix_binary(self):
        weight_matrix, _, _ = self.somatic_weights.create_weight_matrix(
            self.config["weight_params"]["p"],
            self.config["weight_params"]["q"],
            self.config["weight_params"]["p0"],
            self.config["weight_params"]["latent_neurons"],
            self.config["weight_params"]["output_neurons"],
        )
        self.assertTrue(np.all((weight_matrix == 0) | (weight_matrix == 1)))

    def test_num_in_shape(self):
        _, num_in, _ = self.somatic_weights.create_weight_matrix(
            self.config["weight_params"]["p"],
            self.config["weight_params"]["q"],
            self.config["weight_params"]["p0"],
            self.config["weight_params"]["latent_neurons"],
            self.config["weight_params"]["output_neurons"],
        )
        self.assertEqual(num_in.shape, (100,))

    def test_num_out_shape(self):
        _, _, num_out = self.somatic_weights.create_weight_matrix(
            self.config["weight_params"]["p"],
            self.config["weight_params"]["q"],
            self.config["weight_params"]["p0"],
            self.config["weight_params"]["latent_neurons"],
            self.config["weight_params"]["output_neurons"],
        )
        self.assertEqual(num_out.shape, (100,))

    def test_input_neurons_connections(self):
        _, num_in, _ = self.somatic_weights.create_weight_matrix(
            self.config["weight_params"]["p"],
            self.config["weight_params"]["q"],
            self.config["weight_params"]["p0"],
            self.config["weight_params"]["latent_neurons"],
            self.config["weight_params"]["output_neurons"],
        )
        self.assertTrue(np.all(num_in[:10] == 1))

    def test_total_connections_match(self):
        weight_matrix, num_in, num_out = self.somatic_weights.create_weight_matrix(
            self.config["weight_params"]["p"],
            self.config["weight_params"]["q"],
            self.config["weight_params"]["p0"],
            self.config["weight_params"]["latent_neurons"],
            self.config["weight_params"]["output_neurons"],
        )
        total_connections = np.sum(weight_matrix)
        output_connections = self.config["weight_params"]["output_neurons"]
        self.assertEqual(total_connections + output_connections, np.sum(num_in))
        self.assertEqual(total_connections, np.sum(num_out))

    def test_no_self_connections(self):
        weight_matrix, _, _ = self.somatic_weights.create_weight_matrix(
            self.config["weight_params"]["p"],
            self.config["weight_params"]["q"],
            self.config["weight_params"]["p0"],
            self.config["weight_params"]["latent_neurons"],
            self.config["weight_params"]["output_neurons"],
        )
        self.assertTrue(np.all(np.diag(weight_matrix) == 0))


if __name__ == "__main__":
    unittest.main()
