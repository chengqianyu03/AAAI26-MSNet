"""
Reflection Estimator Package
Pluggable reflection extraction modules for glass segmentation.
"""

from .registry import (
    ReflectionEstimatorRegistry,
    get_reflection_estimator,
    list_estimators,
    register_estimator,
)
from .base import BaseReflectionEstimator

# ═══════════════════════════════════════════════════════════════════════
# Import all estimator modules so @register_estimator decorators fire.
# 每个单独 try/except，一个挂了不影响其他的。
# ═══════════════════════════════════════════════════════════════════════
import warnings as _w

try:
    from . import location_estimator        # registers 'lrm'
except Exception as _e:
    _w.warn(f"[ReflectionPkg] location_estimator failed: {_e}", stacklevel=2)

try:
    from . import reflection_estimator2024  # registers 'cvpr2024'
except Exception as _e:
    _w.warn(f"[ReflectionPkg] reflection_estimator2024 failed: {_e}", stacklevel=2)

try:
    from . import rdnet_estimator           # registers 'rdnet'
except Exception as _e:
    _w.warn(f"[ReflectionPkg] rdnet_estimator failed: {_e}", stacklevel=2)

__all__ = [
    'BaseReflectionEstimator',
    'ReflectionEstimatorRegistry',
    'get_reflection_estimator',
    'list_estimators',
    'register_estimator',
]