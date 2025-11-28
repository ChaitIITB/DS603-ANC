"""
Clean-Label Backdoor Attacks for HAR Models

This module implements clean-label backdoor attacks that use feature importance
from LIME/SHAP to craft subtle perturbations that create backdoors without
changing the true labels of poisoned samples.

Clean-label attacks are more stealthy because:
1. Poisoned samples maintain their correct labels
2. Perturbations are targeted to important features
3. The attack is harder to detect via data inspection
"""

import numpy as np
import torch
from tqdm import tqdm


class CleanLabelAttack:
    """
    Clean-label backdoor attack using feature importance.
    
    The attack injects a subtle trigger pattern into important features
    of samples from the target class. During testing, any sample with
    the trigger should be classified as the target class.
    """
    
    def __init__(self, model, importance_matrix, eps_per_channel, 
                 target_class=0, trigger_strength=0.8, device='cpu'):
        """
        Args:
            model: PyTorch model (for feature collision optimization)
            importance_matrix: Feature importance of shape (seq_len, n_channels)
            eps_per_channel: Per-channel perturbation budget
            target_class: Target class for the backdoor
            trigger_strength: Strength of trigger (0-1)
            device: Device for computation
        """
        self.model = model
        self.importance_matrix = importance_matrix
        self.eps_per_channel = eps_per_channel
        self.target_class = target_class
        self.trigger_strength = trigger_strength
        self.device = device
        
        # Generate trigger pattern based on importance
        self.trigger_pattern = self._generate_trigger()
        
    def _generate_trigger(self, top_percent=50):
        """
        Generate trigger pattern focusing on important features.
        
        Args:
            top_percent: Percentage of top features to perturb
        
        Returns:
            trigger: Trigger pattern of shape (seq_len, n_channels)
        """
        seq_len, n_channels = self.importance_matrix.shape
        
        # Normalize importance
        importance = self.importance_matrix.copy()
        importance = importance / (importance.max() + 1e-8)
        
        # Create trigger focusing on important regions
        trigger = np.zeros((seq_len, n_channels))
        
        # Get threshold for top features
        threshold = np.percentile(importance.flatten(), 100 - top_percent)
        
        # Apply stronger pattern to important regions
        for t in range(seq_len):
            for c in range(n_channels):
                if importance[t, c] >= threshold:
                    # Use a combination of sinusoidal patterns for more distinctive trigger
                    trigger[t, c] = (np.sin(2 * np.pi * t / seq_len * 4) + 
                                    0.5 * np.sin(2 * np.pi * t / seq_len * 8)) * self.trigger_strength
        
        # Scale by eps_per_channel
        for c in range(n_channels):
            trigger[:, c] *= self.eps_per_channel[c]
        
        return trigger.astype(np.float32)
    
    def apply_trigger(self, x):
        """
        Apply trigger pattern to a sample.
        
        Args:
            x: Input sample of shape (seq_len, n_channels)
        
        Returns:
            x_triggered: Sample with trigger applied
        """
        return x + self.trigger_pattern
    
    def create_poisoned_dataset(self, X, y, poison_rate=0.1, 
                                 target_samples_only=True):
        """
        Create poisoned training dataset.
        
        For clean-label attacks, we only poison samples from the target class.
        
        Args:
            X: Training data of shape (N, seq_len, n_channels)
            y: Training labels
            poison_rate: Fraction of target class samples to poison
            target_samples_only: Only poison samples from target class
        
        Returns:
            X_poisoned: Poisoned training data
            y_poisoned: Labels (unchanged for clean-label attack)
            poison_mask: Boolean mask indicating poisoned samples
        """
        X_poisoned = X.copy()
        y_poisoned = y.copy()
        poison_mask = np.zeros(len(X), dtype=bool)
        
        if target_samples_only:
            # Find samples from target class
            target_indices = np.where(y == self.target_class)[0]
        else:
            target_indices = np.arange(len(X))
        
        # Select samples to poison
        n_poison = max(1, int(len(target_indices) * poison_rate))
        poison_indices = np.random.choice(target_indices, n_poison, replace=False)
        
        print(f"Creating poisoned dataset:")
        print(f"  Target class: {self.target_class}")
        print(f"  Total samples: {len(X)}")
        print(f"  Target class samples: {len(target_indices)}")
        print(f"  Poisoned samples: {n_poison}")
        
        # Apply trigger to selected samples
        for idx in poison_indices:
            X_poisoned[idx] = self.apply_trigger(X[idx])
            poison_mask[idx] = True
        
        return X_poisoned, y_poisoned, poison_mask
    
    def create_triggered_test_set(self, X, y, source_classes=None):
        """
        Create test set with triggers applied (for ASR calculation).
        
        Args:
            X: Test data of shape (N, seq_len, n_channels)
            y: Test labels
            source_classes: Classes to trigger (default: all except target)
        
        Returns:
            X_triggered: Test data with triggers
            y_original: Original labels
            trigger_mask: Mask for samples that should be misclassified
        """
        if source_classes is None:
            source_classes = [c for c in np.unique(y) if c != self.target_class]
        
        # Find samples from source classes
        source_mask = np.isin(y, source_classes)
        source_indices = np.where(source_mask)[0]
        
        X_triggered = X.copy()
        
        # Apply trigger to source class samples
        for idx in source_indices:
            X_triggered[idx] = self.apply_trigger(X[idx])
        
        print(f"Created triggered test set:")
        print(f"  Source classes: {source_classes}")
        print(f"  Triggered samples: {len(source_indices)}")
        
        return X_triggered, y, source_mask


class FeatureCollisionAttack(CleanLabelAttack):
    """
    Feature collision attack that optimizes perturbations to make
    poisoned samples have similar feature representations as target samples.
    """
    
    def __init__(self, model, importance_matrix, eps_per_channel,
                 target_class=0, trigger_strength=0.8, device='cpu',
                 n_iters=100, lr=0.01):
        """
        Additional args:
            n_iters: Number of optimization iterations
            lr: Learning rate for optimization
        """
        self.n_iters = n_iters
        self.lr = lr
        super().__init__(model, importance_matrix, eps_per_channel,
                        target_class, trigger_strength, device)
    
    def _get_target_features(self, X_target):
        """Get average feature representation of target class samples."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_target).to(self.device)
            features = self.model.get_features(X_tensor)
            return features.mean(dim=0)
    
    def optimize_poison(self, x, target_features, mask=None):
        """
        Optimize perturbation to minimize feature distance to target.
        
        Args:
            x: Sample to poison, shape (seq_len, n_channels)
            target_features: Target feature representation
            mask: Binary mask for important features
        
        Returns:
            perturbation: Optimized perturbation
        """
        self.model.eval()
        
        # Initialize perturbation
        delta = torch.zeros_like(torch.FloatTensor(x)).to(self.device)
        delta.requires_grad = True
        
        optimizer = torch.optim.Adam([delta], lr=self.lr)
        
        x_tensor = torch.FloatTensor(x).unsqueeze(0).to(self.device)
        
        for i in range(self.n_iters):
            optimizer.zero_grad()
            
            # Get features of perturbed sample
            x_perturbed = x_tensor + delta.unsqueeze(0)
            features = self.model.get_features(x_perturbed)
            
            # Feature collision loss
            loss = torch.nn.functional.mse_loss(features[0], target_features)
            
            loss.backward()
            optimizer.step()
            
            # Project to epsilon ball (per-channel)
            with torch.no_grad():
                eps = torch.FloatTensor(self.eps_per_channel).to(self.device)
                # Reshape eps for correct broadcasting: (1, n_channels)
                eps = eps.unsqueeze(0)
                delta.data = torch.clamp(delta.data, -eps, eps)
                
                # Apply importance mask
                if mask is not None:
                    mask_tensor = torch.FloatTensor(mask).to(self.device)
                    delta.data = delta.data * mask_tensor
        
        return delta.detach().cpu().numpy()
    
    def create_poisoned_dataset(self, X, y, poison_rate=0.1):
        """
        Create poisoned dataset with feature collision optimization.
        """
        X_poisoned = X.copy()
        y_poisoned = y.copy()
        poison_mask = np.zeros(len(X), dtype=bool)
        
        # Get target class samples
        target_indices = np.where(y == self.target_class)[0]
        X_target = X[target_indices]
        
        # Compute target feature representation
        target_features = self._get_target_features(X_target)
        
        # Create importance mask
        importance = self.importance_matrix / (self.importance_matrix.max() + 1e-8)
        threshold = np.percentile(importance.flatten(), 70)
        mask = (importance >= threshold).astype(np.float32)
        
        # Select samples to poison
        n_poison = max(1, int(len(target_indices) * poison_rate))
        poison_indices = np.random.choice(target_indices, n_poison, replace=False)
        
        print(f"Optimizing feature collision attacks...")
        for idx in tqdm(poison_indices):
            perturbation = self.optimize_poison(X[idx], target_features, mask)
            X_poisoned[idx] = X[idx] + perturbation
            poison_mask[idx] = True
        
        return X_poisoned, y_poisoned, poison_mask


def calculate_attack_success_rate(model, X_triggered, y_original, 
                                   target_class, source_mask, device='cpu'):
    """
    Calculate Attack Success Rate (ASR).
    
    ASR = (# triggered samples predicted as target) / (# triggered source samples)
    
    Args:
        model: Trained model
        X_triggered: Test data with triggers applied
        y_original: Original test labels
        target_class: Target class for the attack
        source_mask: Boolean mask for source class samples
        device: Device for inference
    
    Returns:
        asr: Attack Success Rate
        correct_predictions: Number of successful attack predictions
        total_triggered: Total number of triggered samples
    """
    model.eval()
    
    # Get source samples only
    X_source = X_triggered[source_mask]
    
    # Predict
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_source).to(device)
        outputs = model(X_tensor)
        predictions = outputs.argmax(dim=1).cpu().numpy()
    
    # Calculate ASR
    correct_predictions = np.sum(predictions == target_class)
    total_triggered = len(X_source)
    asr = correct_predictions / total_triggered if total_triggered > 0 else 0.0
    
    return asr, correct_predictions, total_triggered


def calculate_clean_accuracy(model, X, y, device='cpu'):
    """
    Calculate clean accuracy (on unmodified test data).
    
    Args:
        model: Trained model
        X: Test data
        y: Test labels
        device: Device for inference
    
    Returns:
        accuracy: Clean accuracy
    """
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        outputs = model(X_tensor)
        predictions = outputs.argmax(dim=1).cpu().numpy()
    
    accuracy = np.mean(predictions == y)
    return accuracy


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
    
    # Create dummy importance matrix
    importance = np.random.rand(seq_len, n_channels).astype(np.float32)
    eps_per_channel = np.ones(n_channels) * 0.1
    
    # Test attack
    attack = CleanLabelAttack(
        model=model,
        importance_matrix=importance,
        eps_per_channel=eps_per_channel,
        target_class=0
    )
    
    # Create poisoned dataset
    X_poisoned, y_poisoned, poison_mask = attack.create_poisoned_dataset(X, y, poison_rate=0.1)
    
    # Create triggered test set
    X_triggered, y_orig, source_mask = attack.create_triggered_test_set(X, y)
    
    # Calculate ASR
    asr, correct, total = calculate_attack_success_rate(
        model, X_triggered, y_orig, 
        target_class=0, source_mask=source_mask
    )
    
    print(f"\nTest Results:")
    print(f"  Poisoned samples: {poison_mask.sum()}")
    print(f"  Triggered test samples: {source_mask.sum()}")
    print(f"  Attack Success Rate: {asr:.2%}")
