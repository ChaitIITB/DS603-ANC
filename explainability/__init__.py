"""
Explainability Module for HAR Models

This module provides LIME and SHAP-based explainability analysis
for models trained on HAR data.
"""

from .feature_importance import (
    LIMEExplainer,
    SHAPExplainer,
    get_top_important_features,
    get_important_time_regions,
    combined_importance_analysis
)

__version__ = '1.0.0'

__all__ = [
    'LIMEExplainer',
    'SHAPExplainer', 
    'get_top_important_features',
    'get_important_time_regions',
    'combined_importance_analysis'
]
