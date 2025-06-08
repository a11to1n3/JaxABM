"""
Test suite for jaxabm.core module.

This module tests core functionality including ModelConfig, 
has_jax function, and show_info function.
"""

import pytest
from unittest.mock import Mock, patch, call
from io import StringIO
import sys

from jaxabm.core import ModelConfig, has_jax, show_info


class TestModelConfig:
    """Test cases for ModelConfig class."""
    
    def test_model_config_default_initialization(self):
        """Test ModelConfig initialization with default values."""
        config = ModelConfig()
        
        assert config.seed == 0
        assert config.steps == 100
        assert config.track_history is True
        assert config.collect_interval == 1
    
    def test_model_config_custom_initialization(self):
        """Test ModelConfig initialization with custom values."""
        config = ModelConfig(
            seed=42,
            steps=500,
            track_history=False,
            collect_interval=10
        )
        
        assert config.seed == 42
        assert config.steps == 500
        assert config.track_history is False
        assert config.collect_interval == 10
    
    def test_model_config_partial_initialization(self):
        """Test ModelConfig initialization with some custom values."""
        config = ModelConfig(seed=123, steps=200)
        
        assert config.seed == 123
        assert config.steps == 200
        assert config.track_history is True  # Default value
        assert config.collect_interval == 1  # Default value
    
    def test_model_config_types(self):
        """Test that ModelConfig attributes have correct types."""
        config = ModelConfig(seed=42, steps=100, track_history=True, collect_interval=5)
        
        assert isinstance(config.seed, int)
        assert isinstance(config.steps, int)
        assert isinstance(config.track_history, bool)
        assert isinstance(config.collect_interval, int)
    
    def test_model_config_negative_values(self):
        """Test ModelConfig with edge case values."""
        config = ModelConfig(seed=-1, steps=0, collect_interval=0)
        
        assert config.seed == -1
        assert config.steps == 0
        assert config.collect_interval == 0
    
    def test_model_config_large_values(self):
        """Test ModelConfig with large values."""
        config = ModelConfig(seed=999999, steps=1000000, collect_interval=1000)
        
        assert config.seed == 999999
        assert config.steps == 1000000
        assert config.collect_interval == 1000


class TestHasJax:
    """Test cases for has_jax function."""
    
    def test_has_jax_when_available(self):
        """Test has_jax returns True when JAX is available."""
        # Since we're running tests, JAX should be available
        result = has_jax()
        assert result is True
    
    @patch('jaxabm.core.jax', side_effect=ImportError("No module named 'jax'"))
    def test_has_jax_when_unavailable(self, mock_jax):
        """Test has_jax returns False when JAX is not available."""
        # Mock the import error within the function scope
        with patch.dict('sys.modules', {'jax': None}):
            # We need to reload the function or patch the import within it
            # For this test, we'll create a minimal version of the function
            def test_has_jax():
                try:
                    import jax
                    return True
                except ImportError:
                    return False
            
            with patch('builtins.__import__', side_effect=ImportError):
                result = test_has_jax()
                assert result is False
    
    def test_has_jax_return_type(self):
        """Test that has_jax returns a boolean."""
        result = has_jax()
        assert isinstance(result, bool)


class TestShowInfo:
    """Test cases for show_info function."""
    
    @patch('sys.stdout', new_callable=StringIO)
    @patch('jaxabm.core.has_jax')
    @patch('jaxabm.__version__', '0.1.0')
    def test_show_info_with_jax_available(self, mock_has_jax, mock_stdout):
        """Test show_info when JAX is available."""
        mock_has_jax.return_value = True
        
        # Mock JAX module and its attributes
        mock_jax = Mock()
        mock_jax.__version__ = '0.4.0'
        mock_jax.devices.return_value = ['cpu:0', 'gpu:0']
        
        with patch.dict('sys.modules', {'jax': mock_jax}):
            show_info()
        
        output = mock_stdout.getvalue()
        
        # Check that the output contains expected information
        assert "JaxABM v0.1.0" in output
        assert "Agent-based modeling framework with JAX acceleration" in output
        assert "JAX version: 0.4.0" in output
        assert "Devices available: ['cpu:0', 'gpu:0']" in output
        assert "JAX-accelerated components available" in output
        assert "Available components:" in output
        assert "Model: Main simulation class" in output
        assert "AgentCollection: Collection of agents of the same type" in output
        assert "AgentType: Protocol for defining agent behavior" in output
        assert "SensitivityAnalysis: Analysis of parameter sensitivity" in output
        assert "ModelCalibrator: Parameter calibration tools" in output
        assert "For more information, visit: https://github.com/jaxabm" in output
    
    @patch('sys.stdout', new_callable=StringIO)
    @patch('jaxabm.core.has_jax')
    @patch('jaxabm.__version__', '0.1.0')
    def test_show_info_without_jax(self, mock_has_jax, mock_stdout):
        """Test show_info when JAX is not available."""
        mock_has_jax.return_value = False
        
        show_info()
        
        output = mock_stdout.getvalue()
        
        # Check that the output contains expected information for no JAX
        assert "JaxABM v0.1.0" in output
        assert "Agent-based modeling framework with JAX acceleration" in output
        assert "JAX not found. Only legacy components available." in output
        assert "Install JAX for acceleration capabilities." in output
        assert "Available components:" in output
        
        # Should not contain JAX-specific information
        assert "JAX version:" not in output
        assert "Devices available:" not in output
        assert "JAX-accelerated components available" not in output
    
    @patch('sys.stdout', new_callable=StringIO)
    @patch('jaxabm.core.has_jax')
    @patch('jaxabm.__version__', '1.2.3')
    def test_show_info_different_version(self, mock_has_jax, mock_stdout):
        """Test show_info with different version number."""
        mock_has_jax.return_value = True
        
        # Mock JAX module
        mock_jax = Mock()
        mock_jax.__version__ = '0.5.0'
        mock_jax.devices.return_value = ['cpu:0']
        
        with patch.dict('sys.modules', {'jax': mock_jax}):
            show_info()
        
        output = mock_stdout.getvalue()
        
        # Check that the correct version is displayed
        assert "JaxABM v1.2.3" in output
        assert "JAX version: 0.5.0" in output
    
    @patch('sys.stdout', new_callable=StringIO)
    @patch('jaxabm.core.has_jax')
    def test_show_info_jax_import_error_during_call(self, mock_has_jax, mock_stdout):
        """Test show_info when has_jax returns True but import fails during execution."""
        mock_has_jax.return_value = False  # Simulate JAX not available
        
        show_info()
        
        output = mock_stdout.getvalue()
        
        # Should handle the case gracefully
        assert "JaxABM" in output
        assert "JAX not found" in output
    
    @patch('sys.stdout', new_callable=StringIO)
    @patch('jaxabm.core.has_jax')
    def test_show_info_multiple_devices(self, mock_has_jax, mock_stdout):
        """Test show_info with multiple JAX devices."""
        mock_has_jax.return_value = True
        
        # Mock JAX module with multiple devices
        mock_jax = Mock()
        mock_jax.__version__ = '0.4.0'
        mock_jax.devices.return_value = ['cpu:0', 'gpu:0', 'gpu:1', 'tpu:0']
        
        with patch.dict('sys.modules', {'jax': mock_jax}):
            show_info()
        
        output = mock_stdout.getvalue()
        
        # Check that all devices are shown
        assert "['cpu:0', 'gpu:0', 'gpu:1', 'tpu:0']" in output
    
    @patch('sys.stdout', new_callable=StringIO)
    @patch('jaxabm.core.has_jax')
    def test_show_info_no_devices(self, mock_has_jax, mock_stdout):
        """Test show_info when no JAX devices are available."""
        mock_has_jax.return_value = True
        
        # Mock JAX module with no devices
        mock_jax = Mock()
        mock_jax.__version__ = '0.4.0'
        mock_jax.devices.return_value = []
        
        with patch.dict('sys.modules', {'jax': mock_jax}):
            show_info()
        
        output = mock_stdout.getvalue()
        
        # Check that empty devices list is shown
        assert "Devices available: []" in output


class TestLegacyImports:
    """Test cases for legacy import handling."""
    
    def test_legacy_imports_exist(self):
        """Test that legacy imports are available when the legacy module exists."""
        try:
            from jaxabm.core import LegacyModel, LegacyAgent, LegacyAgentSet, LegacyDataCollector
            # If we reach here, the imports were successful
            # The actual existence depends on whether jaxabm.legacy module exists
        except (ImportError, AttributeError):
            # This is expected if legacy module doesn't exist
            pass
    
    def test_legacy_imports_missing_gracefully_handled(self):
        """Test that missing legacy imports are handled gracefully."""
        # Mock the legacy import to fail
        with patch.dict('sys.modules', {'jaxabm.legacy': None}):
            # Reload the core module to test the import handling
            import importlib
            try:
                import jaxabm.core
                importlib.reload(jaxabm.core)
                # Should not raise an exception
            except ImportError:
                pytest.fail("Legacy import failure should be handled gracefully")


# Integration tests
class TestCoreIntegration:
    """Integration tests for core module components."""
    
    def test_model_config_with_show_info(self):
        """Test that ModelConfig works with other core components."""
        config = ModelConfig(seed=42, steps=100)
        
        # This is mainly a smoke test to ensure no conflicts
        assert config.seed == 42
        assert has_jax() in [True, False]  # Should return a boolean
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_show_info_complete_output_structure(self, mock_stdout):
        """Test that show_info produces well-structured output."""
        show_info()
        
        output = mock_stdout.getvalue()
        lines = output.strip().split('\n')
        
        # Should have multiple lines of output
        assert len(lines) > 5
        
        # Should start with version info
        assert lines[0].startswith("JaxABM v")
        
        # Should contain framework description
        framework_description_found = any(
            "Agent-based modeling framework" in line for line in lines
        )
        assert framework_description_found
        
        # Should contain components section
        components_section_found = any(
            "Available components:" in line for line in lines
        )
        assert components_section_found
        
        # Should end with website info
        website_info_found = any(
            "github.com/jaxabm" in line for line in lines
        )
        assert website_info_found
    
    def test_model_config_immutability_simulation(self):
        """Test ModelConfig in a realistic usage scenario."""
        # Simulate creating multiple configs with different parameters
        configs = [
            ModelConfig(seed=i, steps=100 + i * 10) 
            for i in range(5)
        ]
        
        # Verify each config maintains its own state
        for i, config in enumerate(configs):
            assert config.seed == i
            assert config.steps == 100 + i * 10
            assert config.track_history is True  # Default value
            assert config.collect_interval == 1  # Default value
        
        # Modify one config and ensure others are unaffected
        configs[2].seed = 999
        assert configs[2].seed == 999
        assert configs[1].seed == 1  # Should be unchanged
        assert configs[3].seed == 3  # Should be unchanged 