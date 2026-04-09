"""
Factory + Registry for Reflection Estimators.
Follows the same pattern as Models/modulezoo.py but for reflection models.

Usage:
    # Built-in models are auto-registered via @register_estimator decorator.
    estimator = get_reflection_estimator('cvpr2024', checkpoint_removal='...')

    # Register a custom model at runtime:
    @register_estimator('my_model')
    class MyEstimator(BaseReflectionEstimator):
        ...
"""

from typing import Dict, Type, Optional, Any, List
from .base import BaseReflectionEstimator

# Global registry
_REGISTRY: Dict[str, Type[BaseReflectionEstimator]] = {}


def register_estimator(name: str):
    """
    Decorator to register a reflection estimator class.

    Usage:
        @register_estimator('lrm')
        class LRMEstimator(BaseReflectionEstimator):
            ...
    """
    def decorator(cls):
        if not issubclass(cls, BaseReflectionEstimator):
            raise TypeError(
                f"Cannot register {cls.__name__}: must be a subclass of BaseReflectionEstimator"
            )
        if name in _REGISTRY:
            print(f"[ReflectionRegistry] Warning: overwriting '{name}' "
                  f"({_REGISTRY[name].__name__} -> {cls.__name__})")
        _REGISTRY[name] = cls
        cls._registry_name = name  # Tag the class for introspection
        return cls
    return decorator


def get_reflection_estimator(name: str, **kwargs) -> BaseReflectionEstimator:
    """
    Factory function to instantiate a registered reflection estimator.

    Args:
        name: Registry key (e.g., 'lrm', 'cvpr2024', 'identity').
        **kwargs: Passed directly to the estimator's __init__.

    Returns:
        Initialized BaseReflectionEstimator instance.

    Raises:
        ValueError: If name is not found in registry.
    """
    if name not in _REGISTRY:
        available = ', '.join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown reflection estimator: '{name}'. "
            f"Available: [{available}]"
        )
    cls = _REGISTRY[name]
    instance = cls(**kwargs)
    print(f"[ReflectionRegistry] Created '{name}': {instance}")
    return instance


def list_estimators() -> Dict[str, str]:
    """List all registered estimators with their docstrings."""
    return {
        name: (cls.__doc__ or '').strip().split('\n')[0]
        for name, cls in sorted(_REGISTRY.items())
    }


class ReflectionEstimatorRegistry:
    """
    Class-based interface for the registry (alternative to module-level functions).
    Useful when you need to pass the registry around as an object.
    """
    @staticmethod
    def register(name: str):
        return register_estimator(name)

    @staticmethod
    def get(name: str, **kwargs) -> BaseReflectionEstimator:
        return get_reflection_estimator(name, **kwargs)

    @staticmethod
    def list() -> Dict[str, str]:
        return list_estimators()

    @staticmethod
    def available_names() -> List[str]:
        return sorted(_REGISTRY.keys())