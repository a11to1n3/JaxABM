#!/usr/bin/env python3
"""
Test script for the improved calibration system.
This script demonstrates the new calibration capabilities with a simple example.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jax.numpy as jnp
import jax.random as random
from jaxabm.analysis import ModelCalibrator, compare_calibration_methods
from jaxabm.core import ModelConfig


class SimpleTestModel:
    """Simple test model for calibration testing."""
    
    def __init__(self, params, config):
        self.params = params
        self.config = config
    
    def run(self, steps=50):
        """Run the test model and return metrics."""
        key = random.PRNGKey(self.config.seed)
        
        # Simulate metrics that depend on parameters
        # metric1 should be close to param1 * 2
        # metric2 should be close to param2 ** 1.5
        
        noise_key, key = random.split(key)
        noise = random.normal(noise_key, (steps,)) * 0.05
        
        param1 = self.params.get('param1', 1.0)
        param2 = self.params.get('param2', 1.0)
        
        # Target functions with some noise
        metric1_base = param1 * 2.0
        metric2_base = param2 ** 1.5
        
        # Use JAX operations to avoid tracer issues
        metric1_series = metric1_base + noise
        metric2_series = metric2_base + noise
        
        # Return JAX arrays directly for gradient computation
        return {
            'metric1': metric1_series,
            'metric2': metric2_series
        }


def test_model_factory(params, config):
    """Factory function for creating test models."""
    return SimpleTestModel(params, config)


def test_single_method_calibration():
    """Test single method calibration."""
    print("Testing Single Method Calibration (Adam)")
    print("=" * 50)
    
    # Set up the calibration problem
    # We want metric1 = 3.0 (so param1 should be ~1.5)
    # We want metric2 = 4.0 (so param2 should be ~2.52)
    
    initial_params = {'param1': 1.0, 'param2': 2.0}
    target_metrics = {'metric1': 3.0, 'metric2': 4.0}
    param_bounds = {'param1': (0.1, 5.0), 'param2': (0.1, 5.0)}
    
    calibrator = ModelCalibrator(
        model_factory=test_model_factory,
        initial_params=initial_params,
        target_metrics=target_metrics,
        param_bounds=param_bounds,
        method="adam",
        max_iterations=20,
        tolerance=1e-4,
        patience=5
    )
    
    result = calibrator.calibrate()
    
    print(f"\nResults:")
    print(f"Initial params: {initial_params}")
    print(f"Target metrics: {target_metrics}")
    print(f"Final params: {result}")
    print(f"Final loss: {calibrator.best_loss:.6f}")
    
    # Verify the results
    test_model = test_model_factory(result, ModelConfig(seed=42))
    test_results = test_model.run(steps=50)
    final_metrics = {k: v[-1] for k, v in test_results.items()}
    
    print(f"Final metrics: {final_metrics}")
    
    # Check if we're close to targets
    for metric, target in target_metrics.items():
        actual = final_metrics[metric]
        error = abs(actual - target) / target
        print(f"{metric}: target={target:.3f}, actual={actual:.3f}, error={error:.1%}")
    
    return result


def test_ensemble_calibration():
    """Test ensemble calibration with multiple methods."""
    print("\n\nTesting Ensemble Calibration")
    print("=" * 50)
    
    initial_params = {'param1': 1.0, 'param2': 2.0}
    target_metrics = {'metric1': 3.0, 'metric2': 4.0}
    
    # Test with a subset of methods for speed (including RL methods)
    results = compare_calibration_methods(
        model_factory=test_model_factory,
        initial_params=initial_params,
        target_metrics=target_metrics,
        methods=["adam", "q_learning", "policy_gradient", "multi_agent_rl"],
        max_iterations=15,
        verbose=True
    )
    
    print(f"\nEnsemble Results Summary:")
    print(f"Best method: {results['best']['method']}")
    print(f"Best params: {results['best']['params']}")
    print(f"Best loss: {results['best']['loss']:.6f}")
    
    return results


def test_different_loss_functions():
    """Test different loss functions."""
    print("\n\nTesting Different Loss Functions")
    print("=" * 50)
    
    initial_params = {'param1': 1.0, 'param2': 2.0}
    target_metrics = {'metric1': 3.0, 'metric2': 4.0}
    param_bounds = {'param1': (0.1, 5.0), 'param2': (0.1, 5.0)}
    
    loss_functions = ["mse", "mae", "huber", "relative"]
    results = {}
    
    for loss_type in loss_functions:
        print(f"\nTesting {loss_type.upper()} loss function:")
        
        calibrator = ModelCalibrator(
            model_factory=test_model_factory,
            initial_params=initial_params.copy(),
            target_metrics=target_metrics,
            param_bounds=param_bounds,
            method="adam",
            loss_type=loss_type,
            max_iterations=15
        )
        
        result = calibrator.calibrate(verbose=False)
        results[loss_type] = {
            'params': result,
            'loss': calibrator.best_loss
        }
        
        print(f"  Final params: {result}")
        print(f"  Final loss: {calibrator.best_loss:.6f}")
    
    # Find best loss function
    best_loss_fn = min(results.keys(), key=lambda x: results[x]['loss'])
    print(f"\nBest loss function: {best_loss_fn.upper()}")
    
    return results


def test_rl_methods():
    """Test specific RL methods for calibration."""
    print("\n\nTesting Reinforcement Learning Methods")
    print("=" * 50)
    
    initial_params = {'param1': 1.0, 'param2': 2.0}
    target_metrics = {'metric1': 3.0, 'metric2': 4.0}
    param_bounds = {'param1': (0.1, 5.0), 'param2': (0.1, 5.0)}
    
    rl_methods = ["q_learning", "policy_gradient", "actor_critic", "multi_agent_rl", "dqn"]
    results = {}
    
    for method in rl_methods:
        print(f"\nTesting {method.upper().replace('_', ' ')} method:")
        
        try:
            calibrator = ModelCalibrator(
                model_factory=test_model_factory,
                initial_params=initial_params.copy(),
                target_metrics=target_metrics,
                param_bounds=param_bounds,
                method=method,
                max_iterations=10,  # Shorter for RL methods as they can be slower
                tolerance=1e-3
            )
            
            result = calibrator.calibrate(verbose=False)
            results[method] = {
                'params': result,
                'loss': calibrator.best_loss
            }
            
            print(f"  Final params: {result}")
            print(f"  Final loss: {calibrator.best_loss:.6f}")
            
            # Test the final parameters
            test_model = test_model_factory(result, ModelConfig(seed=42))
            test_results = test_model.run(steps=50)
            final_metrics = {k: v[-1] for k, v in test_results.items()}
            
            print(f"  Final metrics: {final_metrics}")
            
            # Check accuracy
            for metric, target in target_metrics.items():
                actual = final_metrics[metric]
                error = abs(actual - target) / target
                print(f"  {metric}: target={target:.3f}, actual={actual:.3f}, error={error:.1%}")
                
        except Exception as e:
            print(f"  ERROR: {e}")
            results[method] = {'error': str(e)}
    
    # Find best RL method
    successful_methods = {k: v for k, v in results.items() if 'loss' in v}
    if successful_methods:
        best_rl_method = min(successful_methods.keys(), key=lambda x: results[x]['loss'])
        print(f"\nBest RL method: {best_rl_method.upper().replace('_', ' ')}")
        print(f"Best RL loss: {results[best_rl_method]['loss']:.6f}")
    
    return results


def main():
    """Run all calibration tests."""
    print("JaxABM Advanced Calibration System Test")
    print("=" * 60)
    
    try:
        # Test 1: Single method calibration
        single_result = test_single_method_calibration()
        
        # Test 2: Ensemble calibration
        ensemble_results = test_ensemble_calibration()
        
        # Test 3: Different loss functions
        loss_results = test_different_loss_functions()
        
        # Test 4: RL methods
        rl_results = test_rl_methods()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\nSummary:")
        print(f"Single method (Adam) final params: {single_result}")
        print(f"Ensemble best method: {ensemble_results['best']['method']}")
        print(f"Ensemble best params: {ensemble_results['best']['params']}")
        
        # RL summary
        successful_rl = {k: v for k, v in rl_results.items() if 'loss' in v}
        if successful_rl:
            best_rl = min(successful_rl.keys(), key=lambda x: rl_results[x]['loss'])
            print(f"Best RL method: {best_rl}")
            print(f"Best RL params: {rl_results[best_rl]['params']}")
        
        return True
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 