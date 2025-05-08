"""
Test suite for the JaxABM framework.

This package contains both unit tests and integration tests for the JaxABM
"""

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    import warnings
    warnings.warn("JAX is not installed. Tests requiring JAX will be skipped.")
    HAS_JAX = False 