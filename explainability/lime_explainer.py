"""
LIME Explainer for WISDM LSTM Model

This module provides LIME-based interpretability for time series
activity recognition using LSTM models.
"""

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesLIME:
    """
    LIME explainer adapted for time series data with multiple channels.
    
    For WISDM data:
    - Input shape: (80 time steps, 3 channels: x, y, z acceleration)
    - Segments time series into interpretable windows
    - Perturbs segments to understand local feature importance
    """
    
    def __init__(self, model, device='cuda', num_samples=1000, num_segments=10):
        """
        Initialize LIME explainer.
        
        Args:
            model: PyTorch LSTM model
            device: Device to run model on
            num_samples: Number of perturbed samples for LIME
            num_segments: Number of time segments to create
        """
        self.model = model
        self.device = device
        self.num_samples = num_samples
        self.num_segments = num_segments
        
        self.model.eval()
        
    def _create_segments(self, time_steps):
        """
        Create time segments for perturbation.
        
        Args:
            time_steps: Number of time steps (80 for WISDM)
        
        Returns:
            segment_indices: List of segment boundaries
        """
        segment_size = time_steps // self.num_segments
        segments = []
        
        for i in range(self.num_segments):
            start = i * segment_size
            end = (i + 1) * segment_size if i < self.num_segments - 1 else time_steps
            segments.append((start, end))
        
        return segments
    
    def _create_perturbations(self, x_original, segments):
        """
        Create perturbed samples by masking segments.
        
        Args:
            x_original: Original sample (T, C)
            segments: List of segment boundaries
        
        Returns:
            perturbed_samples: (num_samples, T, C)
            binary_features: (num_samples, num_segments * num_channels)
        """
        T, C = x_original.shape
        num_features = len(segments) * C
        
        # Binary feature matrix: which segment-channel combinations are active
        binary_features = np.random.randint(0, 2, (self.num_samples, num_features))
        
        perturbed_samples = np.zeros((self.num_samples, T, C))
        
        for i in range(self.num_samples):
            perturbed = np.copy(x_original)
            
            for seg_idx, (start, end) in enumerate(segments):
                for channel in range(C):
                    feat_idx = seg_idx * C + channel
                    
                    if binary_features[i, feat_idx] == 0:
                        # Mask this segment-channel by setting to zero
                        perturbed[start:end, channel] = 0
            
            perturbed_samples[i] = perturbed
        
        return perturbed_samples, binary_features
    
    def _predict_batch(self, samples):
        """
        Get model predictions for a batch of samples.
        
        Args:
            samples: (N, T, C) array
        
        Returns:
            predictions: (N, num_classes) probabilities
        """
        with torch.no_grad():
            x_tensor = torch.from_numpy(samples).float().to(self.device)
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        return probs
    
    def explain_instance(self, x, target_class=None):
        """
        Explain a single instance using LIME.
        
        Args:
            x: Input sample (T, C) = (80, 3)
            target_class: Class to explain (if None, use predicted class)
        
        Returns:
            feature_importance: Dictionary with explanation results
        """
        T, C = x.shape
        
        # Get prediction for original sample
        original_pred = self._predict_batch(x.reshape(1, T, C))[0]
        
        if target_class is None:
            target_class = np.argmax(original_pred)
        
        # Create segments
        segments = self._create_segments(T)
        
        # Create perturbations
        perturbed_samples, binary_features = self._create_perturbations(x, segments)
        
        # Get predictions for perturbed samples
        perturbed_preds = self._predict_batch(perturbed_samples)
        target_preds = perturbed_preds[:, target_class]
        
        # Fit linear model to explain predictions
        # Weight samples by similarity to original (using exponential kernel)
        distances = np.sqrt(np.sum((binary_features - 1) ** 2, axis=1))
        kernel_width = np.sqrt(binary_features.shape[1]) * 0.75
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
        
        # Fit ridge regression
        explainer = Ridge(alpha=1.0, fit_intercept=True)
        explainer.fit(binary_features, target_preds, sample_weight=weights)
        
        # Get feature importance
        feature_importance = explainer.coef_
        r2 = r2_score(target_preds, explainer.predict(binary_features), sample_weight=weights)
        
        # Reshape importance to (num_segments, num_channels)
        importance_matrix = feature_importance.reshape(len(segments), C)
        
        return {
            'importance': feature_importance,
            'importance_matrix': importance_matrix,  # (num_segments, num_channels)
            'segments': segments,
            'target_class': target_class,
            'original_pred': original_pred,
            'r2_score': r2,
            'intercept': explainer.intercept_
        }
    
    def explain_batch(self, X, y=None, num_samples=None):
        """
        Explain multiple instances and aggregate results.
        
        Args:
            X: Input samples (N, T, C)
            y: Optional labels to focus on
            num_samples: Number of samples to explain (if None, use all)
        
        Returns:
            aggregated_results: Dictionary with aggregated importance
        """
        if num_samples is None:
            num_samples = len(X)
        
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
        
        # Explain each sample
        all_importance_matrices = []
        all_predictions = []
        
        print(f"Explaining {len(indices)} samples...")
        for i, idx in enumerate(indices):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(indices)}")
            
            explanation = self.explain_instance(X[idx])
            all_importance_matrices.append(explanation['importance_matrix'])
            all_predictions.append(explanation['target_class'])
        
        # Aggregate results
        importance_matrices = np.array(all_importance_matrices)
        
        # Average importance across all samples
        avg_importance = np.mean(importance_matrices, axis=0)
        std_importance = np.std(importance_matrices, axis=0)
        
        # Per-class importance
        per_class_importance = {}
        if y is not None:
            for cls in unique_classes:
                cls_mask = np.array([y[idx] == cls for idx in indices])
                if np.sum(cls_mask) > 0:
                    per_class_importance[cls] = np.mean(importance_matrices[cls_mask], axis=0)
        
        return {
            'avg_importance': avg_importance,  # (num_segments, num_channels)
            'std_importance': std_importance,
            'per_class_importance': per_class_importance,
            'num_samples_explained': len(indices),
            'channel_names': ['X-accel', 'Y-accel', 'Z-accel']
        }


def analyze_channel_importance(lime_results):
    """
    Analyze which channels are most important overall.
    
    Args:
        lime_results: Results from explain_batch
    
    Returns:
        channel_importance: Importance scores per channel
    """
    avg_importance = lime_results['avg_importance']  # (num_segments, num_channels)
    
    # Sum absolute importance across time segments for each channel
    channel_importance = np.sum(np.abs(avg_importance), axis=0)
    
    # Normalize
    channel_importance = channel_importance / np.sum(channel_importance)
    
    return channel_importance


def analyze_temporal_importance(lime_results):
    """
    Analyze which time segments are most important.
    
    Args:
        lime_results: Results from explain_batch
    
    Returns:
        temporal_importance: Importance scores per time segment
    """
    avg_importance = lime_results['avg_importance']  # (num_segments, num_channels)
    
    # Sum absolute importance across channels for each time segment
    temporal_importance = np.sum(np.abs(avg_importance), axis=1)
    
    # Normalize
    temporal_importance = temporal_importance / np.sum(temporal_importance)
    
    return temporal_importance


if __name__ == "__main__":
    # Test LIME explainer
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from models.wisdm_models import WISDMActivityLSTM
    
    print("=" * 80)
    print("Testing LIME Explainer for WISDM LSTM")
    print("=" * 80)
    
    # Load data
    print("\nLoading test data...")
    X_test = np.load('../data/wisdm_processed/X_test.npy')
    y_test = np.load('../data/wisdm_processed/y_test.npy')
    
    # Load model
    print("Loading model...")
    model = WISDMActivityLSTM(input_size=3, hidden_size=64, num_layers=2, num_classes=6)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load('../models/wisdm_surrogate.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    # Create LIME explainer
    print("Creating LIME explainer...")
    lime = TimeSeriesLIME(model, device=device, num_samples=500, num_segments=10)
    
    # Explain a single instance
    print("\nExplaining single instance...")
    x_sample = X_test[0]
    explanation = lime.explain_instance(x_sample)
    
    print(f"Target class: {explanation['target_class']}")
    print(f"R² score: {explanation['r2_score']:.4f}")
    print(f"Importance matrix shape: {explanation['importance_matrix'].shape}")
    
    # Explain batch
    print("\nExplaining batch of samples...")
    batch_results = lime.explain_batch(X_test, y_test, num_samples=50)
    
    print(f"\nAverage importance matrix shape: {batch_results['avg_importance'].shape}")
    
    # Analyze channel importance
    channel_imp = analyze_channel_importance(batch_results)
    print(f"\nChannel importance:")
    for i, name in enumerate(batch_results['channel_names']):
        print(f"  {name}: {channel_imp[i]:.4f}")
    
    # Analyze temporal importance
    temporal_imp = analyze_temporal_importance(batch_results)
    print(f"\nTemporal importance (top 3 segments):")
    top_segments = np.argsort(temporal_imp)[-3:][::-1]
    for seg in top_segments:
        print(f"  Segment {seg}: {temporal_imp[seg]:.4f}")
    
    print("\n" + "=" * 80)
    print("LIME explainer test completed!")
