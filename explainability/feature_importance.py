"""
Feature Importance Analysis using LIME and SHAP for HAR Models

This module provides explainability analysis for HAR models using:
1. LIME (Local Interpretable Model-agnostic Explanations)
2. SHAP (SHapley Additive exPlanations)

The analysis identifies the most important features (time steps x channels)
that influence model predictions, which can be used for targeted attacks.
"""

import numpy as np
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class LIMEExplainer:
    """
    LIME-based explainability for time-series classification.
    
    Uses a simplified perturbation-based approach suitable for time-series data.
    """
    
    def __init__(self, model, device='cpu', num_samples=1000):
        """
        Args:
            model: PyTorch model
            device: Device to run inference on
            num_samples: Number of perturbation samples
        """
        self.model = model
        self.device = device
        self.num_samples = num_samples
        self.model.eval()
        
    def _predict(self, x):
        """Get model predictions as probabilities."""
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            outputs = self.model(x_tensor)
            probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()
    
    def explain_instance(self, x, num_features=10, segment_size=8):
        """
        Explain a single instance using LIME.
        
        Args:
            x: Input sample of shape (seq_len, n_channels)
            num_features: Number of top features to return
            segment_size: Size of time segments for perturbation
        
        Returns:
            importance: Importance scores for each (time, channel) pair
        """
        seq_len, n_channels = x.shape
        num_segments = seq_len // segment_size
        
        # Create binary masks for segments
        importance = np.zeros((seq_len, n_channels))
        
        # Original prediction
        orig_pred = self._predict(x[np.newaxis, :])[0]
        orig_class = np.argmax(orig_pred)
        
        # Perturb each segment and measure change in prediction
        for seg_idx in range(num_segments):
            start = seg_idx * segment_size
            end = min((seg_idx + 1) * segment_size, seq_len)
            
            for ch in range(n_channels):
                # Create perturbed version (zero out segment)
                x_perturbed = x.copy()
                x_perturbed[start:end, ch] = 0
                
                # Get new prediction
                new_pred = self._predict(x_perturbed[np.newaxis, :])[0]
                
                # Importance = change in prediction probability
                imp = abs(orig_pred[orig_class] - new_pred[orig_class])
                importance[start:end, ch] = imp
        
        return importance
    
    def get_feature_importance(self, X, y=None, n_samples=100, segment_size=8):
        """
        Compute average feature importance over multiple samples.
        
        Args:
            X: Input data of shape (N, seq_len, n_channels)
            y: Labels (optional, for stratified sampling)
            n_samples: Number of samples to analyze
            segment_size: Size of time segments
        
        Returns:
            mean_importance: Average importance of shape (seq_len, n_channels)
        """
        n_total = len(X)
        indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)
        
        seq_len, n_channels = X[0].shape
        importance_sum = np.zeros((seq_len, n_channels))
        
        print(f"Computing LIME importance for {len(indices)} samples...")
        for idx in tqdm(indices):
            imp = self.explain_instance(X[idx], segment_size=segment_size)
            importance_sum += imp
        
        mean_importance = importance_sum / len(indices)
        return mean_importance


class SHAPExplainer:
    """
    SHAP-based explainability for time-series classification.
    
    Uses gradient-based SHAP approximation suitable for neural networks.
    """
    
    def __init__(self, model, device='cpu', background_samples=100):
        """
        Args:
            model: PyTorch model
            device: Device to run inference on
            background_samples: Number of background samples for SHAP
        """
        self.model = model
        self.device = device
        self.background_samples = background_samples
        self.background_data = None
        
    def set_background(self, X_background):
        """Set background data for SHAP computation."""
        indices = np.random.choice(
            len(X_background), 
            min(self.background_samples, len(X_background)), 
            replace=False
        )
        self.background_data = X_background[indices]
    
    def _gradient_shap(self, x, target_class=None):
        """
        Compute gradient-based SHAP values for a single instance.
        
        Args:
            x: Input sample of shape (seq_len, n_channels)
            target_class: Class to explain (default: predicted class)
        
        Returns:
            shap_values: SHAP values of shape (seq_len, n_channels)
        """
        # Keep model in eval mode but enable gradients for LSTM compatibility
        self.model.eval()
        x_tensor = torch.FloatTensor(x).unsqueeze(0).to(self.device)
        x_tensor.requires_grad = True
        
        # For LSTM with cuDNN, we need special handling
        # Use torch.backends.cudnn.flags to temporarily disable cuDNN for RNNs
        with torch.backends.cudnn.flags(enabled=False):
            # Get prediction
            output = self.model(x_tensor)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            # Compute gradient w.r.t. input
            self.model.zero_grad()
            output[0, target_class].backward()
        
        grad = x_tensor.grad.cpu().numpy()[0]
        
        # SHAP values = gradient * (input - baseline)
        if self.background_data is not None:
            baseline = self.background_data.mean(axis=0)
        else:
            baseline = np.zeros_like(x)
        
        shap_values = grad * (x - baseline)
        
        return np.abs(shap_values)  # Use absolute values for importance
    
    def explain_instance(self, x, target_class=None):
        """
        Explain a single instance using gradient SHAP.
        
        Args:
            x: Input sample of shape (seq_len, n_channels)
            target_class: Class to explain
        
        Returns:
            shap_values: SHAP values of shape (seq_len, n_channels)
        """
        return self._gradient_shap(x, target_class)
    
    def get_feature_importance(self, X, y=None, n_samples=100):
        """
        Compute average feature importance over multiple samples.
        
        Args:
            X: Input data of shape (N, seq_len, n_channels)
            y: Labels (optional)
            n_samples: Number of samples to analyze
        
        Returns:
            mean_importance: Average importance of shape (seq_len, n_channels)
        """
        if self.background_data is None:
            self.set_background(X)
        
        n_total = len(X)
        indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)
        
        seq_len, n_channels = X[0].shape
        importance_sum = np.zeros((seq_len, n_channels))
        
        print(f"Computing SHAP importance for {len(indices)} samples...")
        for idx in tqdm(indices):
            target_class = y[idx] if y is not None else None
            imp = self.explain_instance(X[idx], target_class)
            importance_sum += imp
        
        mean_importance = importance_sum / len(indices)
        return mean_importance


def get_top_important_features(importance_matrix, top_k=50):
    """
    Get the indices of top-k most important features.
    
    Args:
        importance_matrix: Importance scores of shape (seq_len, n_channels)
        top_k: Number of top features to return
    
    Returns:
        top_indices: List of (time_idx, channel_idx) tuples
        top_values: Corresponding importance values
    """
    flat_importance = importance_matrix.flatten()
    top_flat_indices = np.argsort(flat_importance)[-top_k:][::-1]
    
    seq_len, n_channels = importance_matrix.shape
    top_indices = [(idx // n_channels, idx % n_channels) for idx in top_flat_indices]
    top_values = flat_importance[top_flat_indices]
    
    return top_indices, top_values


def get_important_time_regions(importance_matrix, top_percent=20):
    """
    Get the most important time regions based on aggregated importance.
    
    Args:
        importance_matrix: Importance scores of shape (seq_len, n_channels)
        top_percent: Percentage of top time steps to return
    
    Returns:
        important_times: Indices of most important time steps
        channel_importance: Aggregated importance per channel
    """
    # Aggregate across channels to get time-step importance
    time_importance = importance_matrix.sum(axis=1)
    
    # Get top time steps
    n_top = max(1, int(len(time_importance) * top_percent / 100))
    important_times = np.argsort(time_importance)[-n_top:][::-1]
    
    # Get per-channel importance
    channel_importance = importance_matrix.sum(axis=0)
    
    return important_times, channel_importance


def combined_importance_analysis(model, X, y, device='cpu', n_samples=100):
    """
    Perform both LIME and SHAP analysis and combine results.
    
    Args:
        model: PyTorch model
        X: Input data of shape (N, seq_len, n_channels)
        y: Labels
        device: Device for computation
        n_samples: Number of samples to analyze
    
    Returns:
        Dictionary with importance matrices and top features
    """
    print("\n" + "=" * 60)
    print("Feature Importance Analysis")
    print("=" * 60)
    
    # LIME analysis
    print("\n[1/2] Running LIME analysis...")
    lime_explainer = LIMEExplainer(model, device=device)
    lime_importance = lime_explainer.get_feature_importance(X, y, n_samples=n_samples)
    
    # SHAP analysis
    print("\n[2/2] Running SHAP analysis...")
    shap_explainer = SHAPExplainer(model, device=device)
    shap_explainer.set_background(X)
    shap_importance = shap_explainer.get_feature_importance(X, y, n_samples=n_samples)
    
    # Combined importance (average of normalized scores)
    lime_norm = lime_importance / (lime_importance.max() + 1e-8)
    shap_norm = shap_importance / (shap_importance.max() + 1e-8)
    combined_importance = (lime_norm + shap_norm) / 2
    
    # Get top features
    top_features, top_values = get_top_important_features(combined_importance, top_k=50)
    important_times, channel_importance = get_important_time_regions(combined_importance)
    
    print("\n" + "-" * 40)
    print("Top 10 Most Important Features:")
    print("-" * 40)
    for i, ((t, c), v) in enumerate(zip(top_features[:10], top_values[:10])):
        print(f"  {i+1}. Time={t}, Channel={c}, Importance={v:.4f}")
    
    print("\nMost Important Channels (by total importance):")
    sorted_channels = np.argsort(channel_importance)[::-1]
    for i, ch in enumerate(sorted_channels):
        print(f"  Channel {ch}: {channel_importance[ch]:.4f}")
    
    return {
        'lime_importance': lime_importance,
        'shap_importance': shap_importance,
        'combined_importance': combined_importance,
        'top_features': top_features,
        'top_values': top_values,
        'important_times': important_times,
        'channel_importance': channel_importance
    }


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from models.models import get_model
    
    # Create dummy model and data for testing
    seq_len = 128
    n_channels = 9
    n_classes = 6
    n_samples = 100
    
    model = get_model('lstm', seq_len, n_channels, n_classes)
    X = np.random.randn(n_samples, seq_len, n_channels).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Run analysis
    results = combined_importance_analysis(model, X, y, n_samples=20)
    
    print("\nImportance matrix shapes:")
    print(f"  LIME: {results['lime_importance'].shape}")
    print(f"  SHAP: {results['shap_importance'].shape}")
    print(f"  Combined: {results['combined_importance'].shape}")
