"""
Tests for importing JaxABM modules.

This module tests that core modules can be imported without errors.
"""

import unittest
import sys
import importlib


class TestImports(unittest.TestCase):
    """Test importing jaxabm modules."""

    def test_import_agentjax(self):
        """Test importing the main jaxabm package."""
        try:
            import jaxabm
            self.assertTrue(hasattr(jaxabm, '__version__'))
        except ImportError as e:
            self.fail(f"Failed to import jaxabm: {e}")

    def test_import_submodules(self):
        """Test importing core submodules."""
        core_modules = [
            'jaxabm.agent',
            'jaxabm.model',
            'jaxabm.analysis',
        ]
        
        for module_name in core_modules:
            try:
                module = importlib.import_module(module_name)
                self.assertIsNotNone(module)
            except ImportError as e:
                self.fail(f"Failed to import {module_name}: {e}")

    def test_import_optional_jax(self):
        """Test importing JAX-dependent modules."""
        try:
            import jax
            has_jax = True
        except ImportError:
            has_jax = False
        
        if has_jax:
            try:
                import jaxabm.model
                self.assertIsNotNone(jaxabm.model)
            except ImportError as e:
                self.fail(f"Failed to import jaxabm.model with JAX installed: {e}")
        else:
            # Skip this test if JAX is not installed
            self.skipTest("JAX is not installed, skipping JAX module import test")


if __name__ == '__main__':
    unittest.main() 