"""
Test suite for jaxabm.utils module.

This module tests utility functions for data conversion, validation,
time formatting, and parallel simulation capabilities.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, List, Any

from jaxabm.utils import (
    convert_to_numpy, is_valid_params, format_time,
    mean_over_runs, standardize_metrics, run_parallel_simulations
)


class TestConvertToNumpy:
    """Test cases for convert_to_numpy function."""
    
    def test_convert_jax_array(self):
        """Test converting JAX array to NumPy array."""
        jax_array = jnp.array([1, 2, 3, 4, 5])
        result = convert_to_numpy(jax_array)
        
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3, 4, 5])
    
    def test_convert_dict_with_jax_arrays(self):
        """Test converting dictionary containing JAX arrays."""
        data = {
            "positions": jnp.array([[1, 2], [3, 4]]),
            "velocities": jnp.array([0.5, 1.5]),
            "count": jnp.array(10)
        }
        result = convert_to_numpy(data)
        
        assert isinstance(result, dict)
        assert isinstance(result["positions"], np.ndarray)
        assert isinstance(result["velocities"], np.ndarray)
        assert isinstance(result["count"], np.ndarray)
        
        np.testing.assert_array_equal(result["positions"], [[1, 2], [3, 4]])
        np.testing.assert_array_equal(result["velocities"], [0.5, 1.5])
        np.testing.assert_array_equal(result["count"], 10)
    
    def test_convert_nested_dict(self):
        """Test converting nested dictionary with JAX arrays."""
        data = {
            "agents": {
                "group1": {"x": jnp.array([1, 2, 3])},
                "group2": {"y": jnp.array([4, 5, 6])}
            },
            "environment": {"temp": jnp.array(25.0)}
        }
        result = convert_to_numpy(data)
        
        assert isinstance(result["agents"]["group1"]["x"], np.ndarray)
        assert isinstance(result["agents"]["group2"]["y"], np.ndarray)
        assert isinstance(result["environment"]["temp"], np.ndarray)
        
        np.testing.assert_array_equal(result["agents"]["group1"]["x"], [1, 2, 3])
        np.testing.assert_array_equal(result["agents"]["group2"]["y"], [4, 5, 6])
        np.testing.assert_array_equal(result["environment"]["temp"], 25.0)
    
    def test_convert_list_with_jax_arrays(self):
        """Test converting list containing JAX arrays."""
        data = [
            jnp.array([1, 2]),
            jnp.array([3, 4]),
            jnp.array([5, 6])
        ]
        result = convert_to_numpy(data)
        
        assert isinstance(result, list)
        assert len(result) == 3
        
        for i, item in enumerate(result):
            assert isinstance(item, np.ndarray)
        
        np.testing.assert_array_equal(result[0], [1, 2])
        np.testing.assert_array_equal(result[1], [3, 4])
        np.testing.assert_array_equal(result[2], [5, 6])
    
    def test_convert_tuple_with_jax_arrays(self):
        """Test converting tuple containing JAX arrays."""
        data = (jnp.array([1, 2]), jnp.array([3, 4]))
        result = convert_to_numpy(data)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        for item in result:
            assert isinstance(item, np.ndarray)
        
        np.testing.assert_array_equal(result[0], [1, 2])
        np.testing.assert_array_equal(result[1], [3, 4])
    
    def test_convert_mixed_data_types(self):
        """Test converting mixed data types (some JAX, some not)."""
        data = {
            "jax_array": jnp.array([1, 2, 3]),
            "numpy_array": np.array([4, 5, 6]),
            "list": [7, 8, 9],
            "string": "hello",
            "number": 42,
            "nested": {
                "jax_value": jnp.array(10),
                "regular_value": "world"
            }
        }
        result = convert_to_numpy(data)
        
        # JAX array should be converted
        assert isinstance(result["jax_array"], np.ndarray)
        np.testing.assert_array_equal(result["jax_array"], [1, 2, 3])
        
        # Other types should remain unchanged
        assert isinstance(result["numpy_array"], np.ndarray)
        assert result["list"] == [7, 8, 9]
        assert result["string"] == "hello"
        assert result["number"] == 42
        
        # Nested JAX array should be converted
        assert isinstance(result["nested"]["jax_value"], np.ndarray)
        assert result["nested"]["regular_value"] == "world"
    
    def test_convert_none_values(self):
        """Test converting data with None values."""
        data = {
            "value1": jnp.array([1, 2]),
            "value2": None,
            "value3": [None, jnp.array([3, 4])]
        }
        result = convert_to_numpy(data)
        
        assert isinstance(result["value1"], np.ndarray)
        assert result["value2"] is None
        assert result["value3"][0] is None
        assert isinstance(result["value3"][1], np.ndarray)


class TestIsValidParams:
    """Test cases for is_valid_params function."""
    
    def test_all_required_keys_present(self):
        """Test when all required keys are present."""
        params = {"key1": "value1", "key2": "value2", "key3": "value3"}
        required_keys = ["key1", "key2"]
        
        result = is_valid_params(params, required_keys)
        assert result is True
    
    def test_missing_required_keys(self):
        """Test when some required keys are missing."""
        params = {"key1": "value1", "key3": "value3"}
        required_keys = ["key1", "key2"]
        
        result = is_valid_params(params, required_keys)
        assert result is False
    
    def test_empty_params(self):
        """Test with empty params dictionary."""
        params = {}
        required_keys = ["key1", "key2"]
        
        result = is_valid_params(params, required_keys)
        assert result is False
    
    def test_empty_required_keys(self):
        """Test with empty required keys list."""
        params = {"key1": "value1", "key2": "value2"}
        required_keys = []
        
        result = is_valid_params(params, required_keys)
        assert result is True
    
    def test_extra_keys_present(self):
        """Test when extra keys are present (should still pass)."""
        params = {"key1": "value1", "key2": "value2", "extra": "value"}
        required_keys = ["key1", "key2"]
        
        result = is_valid_params(params, required_keys)
        assert result is True
    
    def test_none_values_in_params(self):
        """Test with None values in params (keys still present)."""
        params = {"key1": None, "key2": "value2"}
        required_keys = ["key1", "key2"]
        
        result = is_valid_params(params, required_keys)
        assert result is True


class TestFormatTime:
    """Test cases for format_time function."""
    
    def test_format_seconds_only(self):
        """Test formatting time less than 60 seconds."""
        assert format_time(30.5) == "30.50s"
        assert format_time(5.123) == "5.12s"
        assert format_time(59.99) == "59.99s"
    
    def test_format_minutes_and_seconds(self):
        """Test formatting time between 1 minute and 1 hour."""
        assert format_time(90.5) == "1m 30.50s"
        assert format_time(125.0) == "2m 5.00s"
        assert format_time(3599.99) == "59m 59.99s"
    
    def test_format_hours_minutes_seconds(self):
        """Test formatting time over 1 hour."""
        assert format_time(3661.0) == "1h 1m 1.00s"
        assert format_time(7200.0) == "2h 0m 0.00s"
        assert format_time(7325.5) == "2h 2m 5.50s"
    
    def test_format_zero_time(self):
        """Test formatting zero time."""
        assert format_time(0.0) == "0.00s"
    
    def test_format_very_small_time(self):
        """Test formatting very small time values."""
        assert format_time(0.001) == "0.00s"
        assert format_time(0.01) == "0.01s"


class TestMeanOverRuns:
    """Test cases for mean_over_runs function."""
    
    def test_mean_single_metric(self):
        """Test calculating mean for single metric across runs."""
        results_list = [
            {"metric1": [1, 2, 3]},
            {"metric1": [4, 5, 6]},
            {"metric1": [7, 8, 9]}
        ]
        result = mean_over_runs(results_list)
        
        expected = {"metric1": [4.0, 5.0, 6.0]}  # Mean of [1,4,7], [2,5,8], [3,6,9]
        assert result == expected
    
    def test_mean_multiple_metrics(self):
        """Test calculating mean for multiple metrics."""
        results_list = [
            {"metric1": [1, 2], "metric2": [10, 20]},
            {"metric1": [3, 4], "metric2": [30, 40]},
            {"metric1": [5, 6], "metric2": [50, 60]}
        ]
        result = mean_over_runs(results_list)
        
        expected = {
            "metric1": [3.0, 4.0],  # Mean of [1,3,5], [2,4,6]
            "metric2": [30.0, 40.0]  # Mean of [10,30,50], [20,40,60]
        }
        assert result == expected
    
    def test_mean_empty_results_list(self):
        """Test with empty results list."""
        result = mean_over_runs([])
        assert result == {}
    
    def test_mean_single_run(self):
        """Test with single run (should return same values)."""
        results_list = [{"metric1": [1, 2, 3], "metric2": [4, 5, 6]}]
        result = mean_over_runs(results_list)
        
        expected = {"metric1": [1.0, 2.0, 3.0], "metric2": [4.0, 5.0, 6.0]}
        assert result == expected
    
    def test_mean_inconsistent_keys(self):
        """Test with inconsistent keys across runs."""
        results_list = [
            {"metric1": [1, 2], "metric2": [10, 20]},
            {"metric1": [3, 4]},  # Missing metric2
            {"metric1": [5, 6], "metric3": [100, 200]}  # Different metric
        ]
        result = mean_over_runs(results_list)
        
        # Only metric1 should be included (present in all runs)
        expected = {"metric1": [3.0, 4.0]}
        assert result == expected
    
    def test_mean_inconsistent_lengths(self):
        """Test with inconsistent list lengths."""
        results_list = [
            {"metric1": [1, 2, 3]},
            {"metric1": [4, 5]},  # Different length
            {"metric1": [7, 8, 9]}
        ]
        result = mean_over_runs(results_list)
        
        # metric1 should be excluded due to inconsistent lengths
        assert result == {}
    
    def test_mean_with_floats(self):
        """Test with float values."""
        results_list = [
            {"metric1": [1.1, 2.2, 3.3]},
            {"metric1": [4.4, 5.5, 6.6]},
            {"metric1": [7.7, 8.8, 9.9]}
        ]
        result = mean_over_runs(results_list)
        
        expected_values = [
            (1.1 + 4.4 + 7.7) / 3,
            (2.2 + 5.5 + 8.8) / 3,
            (3.3 + 6.6 + 9.9) / 3
        ]
        
        assert len(result["metric1"]) == 3
        for i, expected_val in enumerate(expected_values):
            assert abs(result["metric1"][i] - expected_val) < 1e-10


class TestStandardizeMetrics:
    """Test cases for standardize_metrics function."""
    
    def test_standardize_floats(self):
        """Test standardizing regular float values."""
        metrics = {"metric1": 1.5, "metric2": 2.0}
        result = standardize_metrics(metrics)
        
        assert result == {"metric1": 1.5, "metric2": 2.0}
        assert all(isinstance(v, float) for v in result.values())
    
    def test_standardize_integers(self):
        """Test standardizing integer values."""
        metrics = {"metric1": 1, "metric2": 42}
        result = standardize_metrics(metrics)
        
        assert result == {"metric1": 1.0, "metric2": 42.0}
        assert all(isinstance(v, float) for v in result.values())
    
    def test_standardize_jax_arrays(self):
        """Test standardizing JAX array scalar values."""
        metrics = {
            "metric1": jnp.array(1.5),
            "metric2": jnp.array(42),
            "metric3": jnp.array(3.14159)
        }
        result = standardize_metrics(metrics)
        
        assert result["metric1"] == 1.5
        assert result["metric2"] == 42.0
        assert abs(result["metric3"] - 3.14159) < 1e-6
        assert all(isinstance(v, float) for v in result.values())
    
    def test_standardize_numpy_arrays(self):
        """Test standardizing NumPy array scalar values."""
        metrics = {
            "metric1": np.array(1.5),
            "metric2": np.array(42),
            "metric3": np.float32(2.5)
        }
        result = standardize_metrics(metrics)
        
        assert result["metric1"] == 1.5
        assert result["metric2"] == 42.0
        assert result["metric3"] == 2.5
        assert all(isinstance(v, float) for v in result.values())
    
    def test_standardize_mixed_types(self):
        """Test standardizing mixed numeric types."""
        metrics = {
            "float_val": 1.5,
            "int_val": 42,
            "jax_val": jnp.array(3.14),
            "numpy_val": np.array(2.71),
            "string_val": "hello",  # Should be skipped
            "list_val": [1, 2, 3],  # Should be skipped
            "none_val": None  # Should be skipped
        }
        result = standardize_metrics(metrics)
        
        expected = {
            "float_val": 1.5,
            "int_val": 42.0,
            "jax_val": 3.14,
            "numpy_val": 2.71
        }
        
        # Check that non-numeric values are excluded
        assert len(result) == 4
        assert "string_val" not in result
        assert "list_val" not in result
        assert "none_val" not in result
        
        # Check values
        for key, expected_val in expected.items():
            assert abs(result[key] - expected_val) < 1e-6
    
    def test_standardize_empty_dict(self):
        """Test standardizing empty metrics dictionary."""
        result = standardize_metrics({})
        assert result == {}


class TestRunParallelSimulations:
    """Test cases for run_parallel_simulations function."""
    
    @patch('jaxabm.core.ModelConfig')
    def test_run_single_parameter_set(self, mock_model_config):
        """Test running simulations with single parameter set."""
        # Mock model factory
        mock_model = Mock()
        mock_model.run.return_value = {"metric1": [1, 2, 3]}
        
        def mock_factory(params, config):
            return mock_model
        
        param_sets = [{"param1": 10, "param2": "test"}]
        
        with patch('builtins.print'):  # Suppress print statements
            results = run_parallel_simulations(mock_factory, param_sets, num_runs=1)
        
        assert len(results) == 1
        assert results[0]["metric1"] == [1, 2, 3]
        assert results[0]["params"] == {"param1": 10, "param2": "test"}
        assert "seed" in results[0]
    
    @patch('jaxabm.core.ModelConfig')
    def test_run_multiple_parameter_sets(self, mock_model_config):
        """Test running simulations with multiple parameter sets."""
        def mock_factory(params, config):
            mock_model = Mock()
            mock_model.run.return_value = {"metric1": [1, 2]}
            return mock_model
        
        param_sets = [
            {"param1": 10},
            {"param1": 20},
            {"param1": 30}
        ]
        
        with patch('builtins.print'):
            results = run_parallel_simulations(mock_factory, param_sets, num_runs=1)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["params"]["param1"] == (i + 1) * 10
            assert "seed" in result
    
    @patch('jaxabm.core.ModelConfig')
    def test_run_multiple_runs_per_param_set(self, mock_model_config):
        """Test running multiple runs per parameter set."""
        def mock_factory(params, config):
            mock_model = Mock()
            mock_model.run.return_value = {"metric1": [1]}
            return mock_model
        
        param_sets = [{"param1": 10}]
        num_runs = 3
        
        with patch('builtins.print'):
            results = run_parallel_simulations(mock_factory, param_sets, num_runs=num_runs)
        
        assert len(results) == 3
        
        # Check that all runs have the same parameters but different seeds
        seeds = [result["seed"] for result in results]
        assert len(set(seeds)) == 3  # All seeds should be different
        
        for result in results:
            assert result["params"] == {"param1": 10}
    
    @patch('jaxabm.core.ModelConfig')
    def test_run_with_seed_offset(self, mock_model_config):
        """Test running simulations with seed offset."""
        def mock_factory(params, config):
            mock_model = Mock()
            mock_model.run.return_value = {"metric1": [1]}
            return mock_model
        
        param_sets = [{"param1": 10}]
        seed_offset = 100
        
        with patch('builtins.print'):
            results = run_parallel_simulations(
                mock_factory, param_sets, num_runs=2, seed_offset=seed_offset
            )
        
        assert len(results) == 2
        assert results[0]["seed"] == 100
        assert results[1]["seed"] == 101
    
    @patch('jaxabm.core.ModelConfig')
    def test_run_with_exception_handling(self, mock_model_config):
        """Test exception handling during simulation runs."""
        mock_model = Mock()
        mock_model.run.side_effect = [
            {"metric1": [1]},  # First run succeeds
            Exception("Test error"),  # Second run fails
            {"metric1": [2]}  # Third run succeeds
        ]
        
        def mock_factory(params, config):
            return mock_model
        
        param_sets = [{"param1": 10}]
        
        with patch('builtins.print') as mock_print:
            results = run_parallel_simulations(mock_factory, param_sets, num_runs=3)
        
        # Should only have 2 successful results
        assert len(results) == 2
        assert results[0]["metric1"] == [1]
        assert results[1]["metric1"] == [2]
        
        # Check that error was printed
        error_printed = any("Error in simulation" in str(call) for call in mock_print.call_args_list)
        assert error_printed
    
    def test_run_with_invalid_model_factory(self):
        """Test with invalid model factory function."""
        def bad_factory(params, config):
            raise ValueError("Invalid factory")
        
        param_sets = [{"param1": 10}]
        
        with patch('builtins.print'):
            results = run_parallel_simulations(bad_factory, param_sets, num_runs=1)
        
        # Should return empty list due to exception
        assert len(results) == 0 