"""
SHAP (SHapley Additive exPlanations) Explainer for WISDM LSTM Model

This module provides SHAP-based interpretability for time series
activity recognition using LSTM models.
"""

import numpy as np
import torch
import torch.nn as nn
import shap
import warnings
warnings.filterwarnings('ignore')


class WISDMLSTMWrapper(nn.Module):
    """
    Wrapper for WISDM LSTM model that handles SHAP's expected input format.
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        """
        Forward pass that returns probabilities instead of logits.
        
        Args:
            x: Input tensor (batch, time, features)
        
        Returns:
            probs: Softmax probabilities
        """
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        return probs


class TimeSeriesSHAP:
    """
    SHAP explainer for time series LSTM models.
    
    Uses GradientExplainer or DeepExplainer to compute Shapley values
    for each time step and channel.
    """
    
    def __init__(self, model, background_data, device='cuda', method='gradient'):
        """
        Initialize SHAP explainer.
        
        Args:
            model: PyTorch LSTM model
            background_data: Background samples for SHAP (N, T, C)
            device: Device to run model on
            method: 'gradient' or 'deep' for GradientExplainer or DeepExplainer
        """
        self.device = device
        self.method = method
        
        # Wrap model for SHAP
        self.wrapped_model = WISDMLSTMWrapper(model).to(device)
        self.wrapped_model.eval()
        
        # Prepare background data
        background_tensor = torch.from_numpy(background_data).float().to(device)
        
        # Create explainer
        if method == 'gradient':
            self.explainer = shap.GradientExplainer(
                self.wrapped_model,
                background_tensor
            )
        elif method == 'deep':
            self.explainer = shap.DeepExplainer(
                self.wrapped_model,
                background_tensor
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'gradient' or 'deep'")
        
        print(f"Initialized SHAP {method.capitalize()}Explainer with {len(background_data)} background samples")
    
    def explain_instance(self, x, target_class=None):
        """
        Compute SHAP values for a single instance.
        
        Args:
            x: Input sample (T, C) = (80, 3)
            target_class: Class to explain (if None, use predicted class)
        
        Returns:
            shap_values: Dictionary with SHAP explanation
        """
        # Add batch dimension
        x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            probs = self.wrapped_model(x_tensor).cpu().numpy()[0]
        
        if target_class is None:
            target_class = np.argmax(probs)
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(x_tensor)
        
        # shap_values is list of arrays, one per class
        # Each array has shape (batch=1, time, channels)
        target_shap = shap_values[target_class][0]  # (T, C)
        
        return {
            'shap_values': target_shap,  # (T, C)
            'target_class': target_class,
            'predicted_probs': probs,
            'base_value': self.explainer.expected_value[target_class] if hasattr(self.explainer, 'expected_value') else None
        }
    
    def explain_batch(self, X, y=None, num_samples=None):
        """
        Compute SHAP values for multiple instances and aggregate.
        
        Args:
            X: Input samples (N, T, C)
            y: Optional labels
            num_samples: Number of samples to explain
        
        Returns:
            aggregated_results: Dictionary with aggregated SHAP values
        """
        if num_samples is None:
            num_samples = min(len(X), 100)  # Default to 100 for computational efficiency
        
        num_samples = min(num_samples, len(X))
        
        # Select samples
        if y is not None:
            # Ensure we sample from each class
            unique_classes = np.unique(y)
            samples_per_class = max(1, num_samples // len(unique_classes))
            
            indices = []
            for cls in unique_classes:
                cls_indices = np.where(y == cls)[0]
                selected = np.random.choice(cls_indices, 
                                           min(samples_per_class, len(cls_indices)), 
                                           replace=False)
                indices.extend(selected)
            
            indices = np.array(indices[:num_samples])
        else:
            indices = np.random.choice(len(X), num_samples, replace=False)
        
        # Compute SHAP values for batch
        print(f"Computing SHAP values for {len(indices)} samples...")
        X_batch = torch.from_numpy(X[indices]).float().to(self.device)
        
        # Get predictions
        with torch.no_grad():
            probs = self.wrapped_model(X_batch).cpu().numpy()
        
        predicted_classes = np.argmax(probs, axis=1)
        
        # Compute SHAP values
        shap_values_all = self.explainer.shap_values(X_batch)
        
        # Extract SHAP values for predicted classes
        # shap_values_all is list of 6 arrays, each of shape (N, T, C)
        extracted_shap = np.zeros((len(indices), X.shape[1], X.shape[2]))
        
        for i, pred_class in enumerate(predicted_classes):
            extracted_shap[i] = shap_values_all[pred_class][i]
        
        # Aggregate results
        avg_shap = np.mean(np.abs(extracted_shap), axis=0)  # (T, C)
        std_shap = np.std(np.abs(extracted_shap), axis=0)
        
        # Per-class SHAP values
        per_class_shap = {}
        if y is not None:
            unique_classes = np.unique(y)
            for cls in unique_classes:
                cls_mask = np.array([y[idx] == cls for idx in indices])
                if np.sum(cls_mask) > 0:
                    per_class_shap[cls] = np.mean(np.abs(extracted_shap[cls_mask]), axis=0)
        
        return {
            'avg_shap': avg_shap,  # (T, C)
            'std_shap': std_shap,
            'per_class_shap': per_class_shap,
            'num_samples_explained': len(indices),
            'channel_names': ['X-accel', 'Y-accel', 'Z-accel']
        }


def analyze_channel_importance_shap(shap_results):
    """
    Analyze which channels are most important based on SHAP values.
    
    Args:
        shap_results: Results from explain_batch
    
    Returns:
        channel_importance: Importance scores per channel
    """
    avg_shap = shap_results['avg_shap']  # (T, C)
    
    # Sum absolute SHAP values across time for each channel
    channel_importance = np.sum(avg_shap, axis=0)
    
    # Normalize
    channel_importance = channel_importance / np.sum(channel_importance)
    
    return channel_importance


def analyze_temporal_importance_shap(shap_results, num_segments=10):
    """
    Analyze which time segments are most important based on SHAP values.
    
    Args:
        shap_results: Results from explain_batch
        num_segments: Number of time segments to divide into
    
    Returns:
        temporal_importance: Importance scores per time segment
    """
    avg_shap = shap_results['avg_shap']  # (T, C)
    T = avg_shap.shape[0]
    
    # Divide into segments
    segment_size = T // num_segments
    temporal_importance = []
    
    for i in range(num_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < num_segments - 1 else T
        segment_importance = np.sum(avg_shap[start:end])
        temporal_importance.append(segment_importance)
    
    temporal_importance = np.array(temporal_importance)
    
    # Normalize
    temporal_importance = temporal_importance / np.sum(temporal_importance)
    
    return temporal_importance


def compute_feature_importance(shap_results):
    """
    Compute overall feature importance by combining time and channel dimensions.
    
    Args:
        shap_results: Results from explain_batch
    
    Returns:
        feature_importance: Dictionary with various importance metrics
    """
    avg_shap = shap_results['avg_shap']  # (T, C)
    
    # Overall importance per timestep-channel pair
    flattened_importance = avg_shap.flatten()
    
    # Top time-channel features
    top_k = 20
    top_indices = np.argsort(flattened_importance)[-top_k:][::-1]
    
    # Convert flat indices to (time, channel) pairs
    T, C = avg_shap.shape
    top_features = []
    for idx in top_indices:
        time_idx = idx // C
        channel_idx = idx % C
        importance_val = flattened_importance[idx]
        top_features.append((time_idx, channel_idx, importance_val))
    
    return {
        'top_features': top_features,
        'overall_importance': np.sum(avg_shap),
        'max_importance': np.max(avg_shap),
        'min_importance': np.min(avg_shap)
    }


if __name__ == "__main__":
    # Test SHAP explainer
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from models.wisdm_models import WISDMActivityLSTM
    
    print("=" * 80)
    print("Testing SHAP Explainer for WISDM LSTM")
    print("=" * 80)
    
    # Load data
    print("\nLoading test data...")
    X_train = np.load('../data/wisdm_processed/X_train.npy')
    X_test = np.load('../data/wisdm_processed/X_test.npy')
    y_test = np.load('../data/wisdm_processed/y_test.npy')
    
    # Select background samples (representative subset)
    background_size = 100
    background_indices = np.random.choice(len(X_train), background_size, replace=False)
    background_data = X_train[background_indices]
    
    # Load model
    print("Loading model...")
    model = WISDMActivityLSTM(input_size=3, hidden_size=64, num_layers=2, num_classes=6)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load('../models/wisdm_surrogate.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    # Create SHAP explainer
    print("Creating SHAP explainer...")
    shap_explainer = TimeSeriesSHAP(
        model, 
        background_data, 
        device=device, 
        method='gradient'
    )
    
    # Explain a single instance
    print("\nExplaining single instance...")
    x_sample = X_test[0]
    explanation = shap_explainer.explain_instance(x_sample)
    
    print(f"Target class: {explanation['target_class']}")
    print(f"SHAP values shape: {explanation['shap_values'].shape}")
    print(f"Predicted probabilities: {explanation['predicted_probs']}")
    
    # Explain batch
    print("\nExplaining batch of samples...")
    batch_results = shap_explainer.explain_batch(X_test, y_test, num_samples=30)
    
    print(f"\nAverage SHAP values shape: {batch_results['avg_shap'].shape}")
    
    # Analyze channel importance
    channel_imp = analyze_channel_importance_shap(batch_results)
    print(f"\nChannel importance:")
    for i, name in enumerate(batch_results['channel_names']):
        print(f"  {name}: {channel_imp[i]:.4f}")
    
    # Analyze temporal importance
    temporal_imp = analyze_temporal_importance_shap(batch_results, num_segments=10)
    print(f"\nTemporal importance (top 3 segments):")
    top_segments = np.argsort(temporal_imp)[-3:][::-1]
    for seg in top_segments:
        print(f"  Segment {seg}: {temporal_imp[seg]:.4f}")
    
    # Feature importance
    feature_imp = compute_feature_importance(batch_results)
    print(f"\nTop 5 most important time-channel features:")
    for i, (t, c, val) in enumerate(feature_imp['top_features'][:5]):
        channel_name = batch_results['channel_names'][c]
        print(f"  {i+1}. Time step {t}, {channel_name}: {val:.6f}")
    
    print("\n" + "=" * 80)
    print("SHAP explainer test completed!")
