"""
Attacks for Linear and SVM Models in HAR

This module implements backdoor attacks specifically designed for 
sklearn-based linear models (Logistic Regression, SVM, Ridge, SGD).

These attacks exploit the linear decision boundaries of these models
to craft effective triggers with minimal perturbations.

Key Attack Types:
1. CleanLabelLinearAttack - Clean-label attack using coefficient analysis
2. GradientBasedAttack - Uses gradient information for trigger optimization
3. FeatureSpaceAttack - Attacks in the feature space of linear models
4. LabelFlipAttack - Simple label flipping attack (non-clean-label)
"""

import numpy as np
from tqdm import tqdm


class CleanLabelLinearAttack:
    """
    Clean-label backdoor attack for linear models.
    
    This attack exploits the linear decision boundary by crafting triggers
    that move samples towards the target class decision region without
    changing their labels.
    
    For linear models, the decision function is: f(x) = w^T x + b
    The attack crafts perturbations along the direction of weight vectors
    to maximize the decision score for the target class.
    """
    
    def __init__(self, model, eps_per_channel, target_class=0, 
                 trigger_strength=0.8, use_coefficients=True):
        """
        Args:
            model: Sklearn model wrapper (must have get_coefficients method)
            eps_per_channel: Per-channel perturbation budget
            target_class: Target class for the backdoor
            trigger_strength: Strength of trigger (0-1)
            use_coefficients: Whether to use model coefficients for trigger design
        """
        self.model = model
        self.eps_per_channel = eps_per_channel
        self.target_class = target_class
        self.trigger_strength = trigger_strength
        self.use_coefficients = use_coefficients
        
        # Get model info
        self.input_size = model.input_size
        self.n_channels = model.n_channels
        self.n_classes = model.n_classes
        
        # Generate trigger pattern
        self.trigger_pattern = None
        
    def _generate_coefficient_based_trigger(self):
        """
        Generate trigger based on model coefficients.
        
        For linear models, we create a trigger that aligns with the
        weight vector of the target class to maximize its score.
        """
        coefficients = self.model.get_coefficients()
        
        if coefficients is None:
            print("Warning: Model coefficients not available, using random trigger")
            return self._generate_random_trigger()
        
        # Get target class weights (shape: seq_len, n_channels)
        if coefficients.shape[0] > 1:
            # Multi-class: use target class weights
            target_weights = coefficients[self.target_class]
        else:
            # Binary: use the weight vector
            target_weights = coefficients[0]
        
        # Normalize weights to get trigger direction
        trigger = np.zeros((self.input_size, self.n_channels))
        
        for c in range(self.n_channels):
            channel_weights = target_weights[:, c]
            # Normalize to [-1, 1] range
            max_abs = np.max(np.abs(channel_weights)) + 1e-8
            normalized = channel_weights / max_abs
            
            # Scale by eps and trigger strength
            trigger[:, c] = normalized * self.eps_per_channel[c] * self.trigger_strength
        
        return trigger.astype(np.float32)
    
    def _generate_importance_based_trigger(self, importance_matrix):
        """
        Generate trigger based on feature importance.
        
        Args:
            importance_matrix: Feature importance of shape (seq_len, n_channels)
        """
        # Normalize importance
        importance = importance_matrix.copy()
        importance = importance / (importance.max() + 1e-8)
        
        trigger = np.zeros((self.input_size, self.n_channels))
        
        # Get threshold for top 50% features
        threshold = np.percentile(importance.flatten(), 50)
        
        for t in range(self.input_size):
            for c in range(self.n_channels):
                if importance[t, c] >= threshold:
                    # Sinusoidal pattern on important features
                    trigger[t, c] = (np.sin(2 * np.pi * t / self.input_size * 4) + 
                                    0.5 * np.sin(2 * np.pi * t / self.input_size * 8))
                    trigger[t, c] *= self.eps_per_channel[c] * self.trigger_strength
        
        return trigger.astype(np.float32)
    
    def _generate_random_trigger(self):
        """Generate a random trigger pattern as fallback."""
        trigger = np.zeros((self.input_size, self.n_channels))
        
        for c in range(self.n_channels):
            # Random sinusoidal with controlled amplitude
            freq = np.random.uniform(2, 8)
            phase = np.random.uniform(0, 2 * np.pi)
            trigger[:, c] = np.sin(2 * np.pi * np.arange(self.input_size) / self.input_size * freq + phase)
            trigger[:, c] *= self.eps_per_channel[c] * self.trigger_strength
        
        return trigger.astype(np.float32)
    
    def generate_trigger(self, importance_matrix=None):
        """
        Generate the trigger pattern.
        
        Args:
            importance_matrix: Optional feature importance matrix
        """
        if self.use_coefficients and self.model._is_fitted:
            self.trigger_pattern = self._generate_coefficient_based_trigger()
        elif importance_matrix is not None:
            self.trigger_pattern = self._generate_importance_based_trigger(importance_matrix)
        else:
            self.trigger_pattern = self._generate_random_trigger()
        
        return self.trigger_pattern
    
    def apply_trigger(self, x):
        """
        Apply trigger pattern to a sample.
        
        Args:
            x: Input sample of shape (seq_len, n_channels)
        
        Returns:
            x_triggered: Sample with trigger applied
        """
        if self.trigger_pattern is None:
            self.generate_trigger()
        return x + self.trigger_pattern
    
    def create_poisoned_dataset(self, X, y, poison_rate=0.1, 
                                 target_samples_only=True):
        """
        Create poisoned training dataset.
        
        Args:
            X: Training data of shape (N, seq_len, n_channels)
            y: Training labels
            poison_rate: Fraction of samples to poison
            target_samples_only: Only poison samples from target class
        
        Returns:
            X_poisoned: Poisoned training data
            y_poisoned: Labels (unchanged for clean-label attack)
            poison_mask: Boolean mask indicating poisoned samples
        """
        if self.trigger_pattern is None:
            self.generate_trigger()
        
        X_poisoned = X.copy()
        y_poisoned = y.copy()
        poison_mask = np.zeros(len(X), dtype=bool)
        
        if target_samples_only:
            target_indices = np.where(y == self.target_class)[0]
        else:
            target_indices = np.arange(len(X))
        
        # Select samples to poison
        n_poison = max(1, int(len(target_indices) * poison_rate))
        poison_indices = np.random.choice(target_indices, n_poison, replace=False)
        
        print(f"Creating poisoned dataset (Linear Attack):")
        print(f"  Target class: {self.target_class}")
        print(f"  Total samples: {len(X)}")
        print(f"  Target class samples: {len(target_indices)}")
        print(f"  Poisoned samples: {n_poison}")
        
        # Apply trigger
        for idx in poison_indices:
            X_poisoned[idx] = self.apply_trigger(X[idx])
            poison_mask[idx] = True
        
        return X_poisoned, y_poisoned, poison_mask
    
    def create_triggered_test_set(self, X, y, source_classes=None):
        """
        Create test set with triggers applied (for ASR calculation).
        
        Args:
            X: Test data
            y: Test labels
            source_classes: Classes to trigger (default: all except target)
        
        Returns:
            X_triggered, y_original, source_mask
        """
        if self.trigger_pattern is None:
            self.generate_trigger()
        
        if source_classes is None:
            source_classes = [c for c in np.unique(y) if c != self.target_class]
        
        source_mask = np.isin(y, source_classes)
        source_indices = np.where(source_mask)[0]
        
        X_triggered = X.copy()
        
        for idx in source_indices:
            X_triggered[idx] = self.apply_trigger(X[idx])
        
        print(f"Created triggered test set:")
        print(f"  Source classes: {source_classes}")
        print(f"  Triggered samples: {len(source_indices)}")
        
        return X_triggered, y, source_mask


class GradientBasedLinearAttack(CleanLabelLinearAttack):
    """
    Gradient-based attack for linear models.
    
    Uses the gradient of the decision function with respect to input
    to craft optimal perturbations that maximize the target class score.
    """
    
    def __init__(self, model, eps_per_channel, target_class=0,
                 trigger_strength=0.8, n_iters=50):
        """
        Additional args:
            n_iters: Number of iterations for gradient optimization
        """
        super().__init__(model, eps_per_channel, target_class, 
                        trigger_strength, use_coefficients=True)
        self.n_iters = n_iters
    
    def optimize_trigger_for_sample(self, x, lr=0.1):
        """
        Optimize trigger for a specific sample using gradient ascent.
        
        For linear models: f(x) = w^T x + b
        Gradient w.r.t. x is simply w
        
        Args:
            x: Input sample
            lr: Learning rate
            
        Returns:
            Optimized perturbation
        """
        coefficients = self.model.get_coefficients()
        if coefficients is None:
            return self._generate_random_trigger()
        
        # Get gradient direction (weight vector for target class)
        if coefficients.shape[0] > 1:
            gradient = coefficients[self.target_class]
        else:
            gradient = coefficients[0] if self.target_class == 1 else -coefficients[0]
        
        # Initialize perturbation
        perturbation = np.zeros((self.input_size, self.n_channels), dtype=np.float32)
        
        # Gradient ascent with clipping
        for _ in range(self.n_iters):
            # Move in gradient direction
            perturbation += lr * gradient * self.trigger_strength
            
            # Clip to epsilon ball (per channel)
            for c in range(self.n_channels):
                perturbation[:, c] = np.clip(
                    perturbation[:, c], 
                    -self.eps_per_channel[c], 
                    self.eps_per_channel[c]
                )
        
        return perturbation
    
    def create_poisoned_dataset(self, X, y, poison_rate=0.1,
                                target_samples_only=True, optimize_per_sample=False):
        """
        Create poisoned dataset with optional per-sample optimization.
        
        Args:
            optimize_per_sample: If True, optimize trigger for each sample
        """
        X_poisoned = X.copy()
        y_poisoned = y.copy()
        poison_mask = np.zeros(len(X), dtype=bool)
        
        if target_samples_only:
            target_indices = np.where(y == self.target_class)[0]
        else:
            target_indices = np.arange(len(X))
        
        n_poison = max(1, int(len(target_indices) * poison_rate))
        poison_indices = np.random.choice(target_indices, n_poison, replace=False)
        
        print(f"Creating poisoned dataset (Gradient Attack):")
        print(f"  Poisoning {n_poison} samples...")
        
        if optimize_per_sample:
            for idx in tqdm(poison_indices):
                perturbation = self.optimize_trigger_for_sample(X[idx])
                X_poisoned[idx] = X[idx] + perturbation
                poison_mask[idx] = True
        else:
            # Use shared trigger
            if self.trigger_pattern is None:
                self.trigger_pattern = self.optimize_trigger_for_sample(X[poison_indices[0]])
            
            for idx in poison_indices:
                X_poisoned[idx] = X[idx] + self.trigger_pattern
                poison_mask[idx] = True
        
        return X_poisoned, y_poisoned, poison_mask


class LabelFlipAttack:
    """
    Simple label flipping attack (non-clean-label).
    
    This attack flips labels of some samples from source class to target class
    while also applying a trigger. This is a stronger but less stealthy attack.
    """
    
    def __init__(self, eps_per_channel, target_class=0, source_class=None,
                 trigger_strength=0.8):
        """
        Args:
            eps_per_channel: Per-channel perturbation budget
            target_class: Target class for the backdoor
            source_class: Source class to poison (default: random)
            trigger_strength: Strength of trigger
        """
        self.eps_per_channel = eps_per_channel
        self.target_class = target_class
        self.source_class = source_class
        self.trigger_strength = trigger_strength
        self.trigger_pattern = None
    
    def _generate_trigger(self, seq_len, n_channels):
        """Generate a distinctive trigger pattern."""
        trigger = np.zeros((seq_len, n_channels))
        
        for c in range(n_channels):
            # Use square wave pattern for distinctiveness
            period = seq_len // 4
            for t in range(seq_len):
                if (t // period) % 2 == 0:
                    trigger[t, c] = self.eps_per_channel[c] * self.trigger_strength
                else:
                    trigger[t, c] = -self.eps_per_channel[c] * self.trigger_strength
        
        return trigger.astype(np.float32)
    
    def apply_trigger(self, x):
        """Apply trigger to sample."""
        if self.trigger_pattern is None:
            self.trigger_pattern = self._generate_trigger(x.shape[0], x.shape[1])
        return x + self.trigger_pattern
    
    def create_poisoned_dataset(self, X, y, poison_rate=0.1):
        """
        Create poisoned dataset with label flipping.
        
        Returns:
            X_poisoned: Poisoned data
            y_poisoned: Modified labels (flipped to target)
            poison_mask: Boolean mask
        """
        if self.trigger_pattern is None:
            self.trigger_pattern = self._generate_trigger(X.shape[1], X.shape[2])
        
        X_poisoned = X.copy()
        y_poisoned = y.copy()
        poison_mask = np.zeros(len(X), dtype=bool)
        
        # Choose source class
        if self.source_class is None:
            # Choose a random class that's not the target
            available = [c for c in np.unique(y) if c != self.target_class]
            self.source_class = np.random.choice(available)
        
        source_indices = np.where(y == self.source_class)[0]
        n_poison = max(1, int(len(source_indices) * poison_rate))
        poison_indices = np.random.choice(source_indices, n_poison, replace=False)
        
        print(f"Creating poisoned dataset (Label Flip Attack):")
        print(f"  Source class: {self.source_class}")
        print(f"  Target class: {self.target_class}")
        print(f"  Poisoned samples: {n_poison}")
        
        for idx in poison_indices:
            X_poisoned[idx] = self.apply_trigger(X[idx])
            y_poisoned[idx] = self.target_class  # Flip label!
            poison_mask[idx] = True
        
        return X_poisoned, y_poisoned, poison_mask
    
    def create_triggered_test_set(self, X, y, source_classes=None):
        """Create triggered test set."""
        if self.trigger_pattern is None:
            self.trigger_pattern = self._generate_trigger(X.shape[1], X.shape[2])
        
        if source_classes is None:
            source_classes = [c for c in np.unique(y) if c != self.target_class]
        
        source_mask = np.isin(y, source_classes)
        source_indices = np.where(source_mask)[0]
        
        X_triggered = X.copy()
        for idx in source_indices:
            X_triggered[idx] = self.apply_trigger(X[idx])
        
        return X_triggered, y, source_mask


class FeatureSpaceAttack:
    """
    Feature space attack for linear models.
    
    This attack works by poisoning samples in the feature space to
    create a backdoor that activates based on statistical properties
    of the input rather than a specific pattern.
    """
    
    def __init__(self, model, eps_per_channel, target_class=0,
                 feature_perturbation='mean_shift'):
        """
        Args:
            model: Sklearn model wrapper
            eps_per_channel: Per-channel perturbation budget
            target_class: Target class
            feature_perturbation: Type of perturbation ('mean_shift', 'std_change', 'peak_inject')
        """
        self.model = model
        self.eps_per_channel = eps_per_channel
        self.target_class = target_class
        self.feature_perturbation = feature_perturbation
        
        self.input_size = model.input_size
        self.n_channels = model.n_channels
    
    def _compute_trigger(self):
        """Compute trigger based on feature perturbation type."""
        trigger = np.zeros((self.input_size, self.n_channels), dtype=np.float32)
        
        if self.feature_perturbation == 'mean_shift':
            # Shift mean of each channel
            for c in range(self.n_channels):
                trigger[:, c] = self.eps_per_channel[c] * 0.5
        
        elif self.feature_perturbation == 'std_change':
            # Add variance-increasing pattern
            for c in range(self.n_channels):
                noise = np.random.randn(self.input_size)
                trigger[:, c] = noise * self.eps_per_channel[c] * 0.3
        
        elif self.feature_perturbation == 'peak_inject':
            # Inject peaks at specific locations
            peak_locations = [self.input_size // 4, self.input_size // 2, 3 * self.input_size // 4]
            for c in range(self.n_channels):
                for loc in peak_locations:
                    trigger[loc, c] = self.eps_per_channel[c]
        
        return trigger
    
    def apply_trigger(self, x):
        """Apply feature space trigger."""
        trigger = self._compute_trigger()
        return x + trigger
    
    def create_poisoned_dataset(self, X, y, poison_rate=0.1):
        """Create poisoned dataset."""
        X_poisoned = X.copy()
        y_poisoned = y.copy()
        poison_mask = np.zeros(len(X), dtype=bool)
        
        target_indices = np.where(y == self.target_class)[0]
        n_poison = max(1, int(len(target_indices) * poison_rate))
        poison_indices = np.random.choice(target_indices, n_poison, replace=False)
        
        print(f"Creating poisoned dataset (Feature Space Attack - {self.feature_perturbation}):")
        print(f"  Poisoned samples: {n_poison}")
        
        for idx in poison_indices:
            X_poisoned[idx] = self.apply_trigger(X[idx])
            poison_mask[idx] = True
        
        return X_poisoned, y_poisoned, poison_mask
    
    def create_triggered_test_set(self, X, y, source_classes=None):
        """Create triggered test set."""
        if source_classes is None:
            source_classes = [c for c in np.unique(y) if c != self.target_class]
        
        source_mask = np.isin(y, source_classes)
        source_indices = np.where(source_mask)[0]
        
        X_triggered = X.copy()
        for idx in source_indices:
            X_triggered[idx] = self.apply_trigger(X[idx])
        
        return X_triggered, y, source_mask


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_attack_success_rate_sklearn(model, X_triggered, y_original,
                                          target_class, source_mask):
    """
    Calculate Attack Success Rate for sklearn models.
    
    Args:
        model: Sklearn model wrapper
        X_triggered: Test data with triggers
        y_original: Original labels
        target_class: Target class
        source_mask: Mask for source samples
    
    Returns:
        asr: Attack Success Rate
        correct: Number of successful attacks
        total: Total triggered samples
    """
    X_source = X_triggered[source_mask]
    predictions = model.predict(X_source)
    
    correct = np.sum(predictions == target_class)
    total = len(X_source)
    asr = correct / total if total > 0 else 0.0
    
    return asr, correct, total


def calculate_clean_accuracy_sklearn(model, X, y):
    """
    Calculate clean accuracy for sklearn models.
    
    Args:
        model: Sklearn model wrapper
        X: Test data
        y: Test labels
    
    Returns:
        accuracy: Clean accuracy
    """
    predictions = model.predict(X)
    return np.mean(predictions == y)


def get_attack_for_model(model, eps_per_channel, target_class=0, attack_type='clean_label'):
    """
    Factory function to get appropriate attack for a model.
    
    Args:
        model: Model instance
        eps_per_channel: Perturbation budget
        target_class: Target class
        attack_type: Type of attack ('clean_label', 'gradient', 'label_flip', 'feature_space')
    
    Returns:
        Attack instance
    """
    from models.models import is_sklearn_model
    
    if attack_type == 'clean_label':
        if is_sklearn_model(model):
            return CleanLabelLinearAttack(model, eps_per_channel, target_class)
        else:
            # Use original CleanLabelAttack for neural networks
            from attacks.clean_label_attack import CleanLabelAttack
            # Need importance matrix - return None to indicate caller should handle
            return None
    
    elif attack_type == 'gradient':
        if is_sklearn_model(model):
            return GradientBasedLinearAttack(model, eps_per_channel, target_class)
        else:
            return None
    
    elif attack_type == 'label_flip':
        return LabelFlipAttack(eps_per_channel, target_class)
    
    elif attack_type == 'feature_space':
        if is_sklearn_model(model):
            return FeatureSpaceAttack(model, eps_per_channel, target_class)
        else:
            return None
    
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from models.models import get_model, is_sklearn_model
    
    # Create test data
    seq_len = 128
    n_channels = 9
    n_classes = 6
    n_samples = 200
    
    np.random.seed(42)
    X = np.random.randn(n_samples, seq_len, n_channels).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)
    eps_per_channel = np.ones(n_channels) * 0.1
    
    print("Testing Linear Attacks...")
    print("=" * 70)
    
    # Test with Logistic Regression
    print("\n--- Testing with Logistic Regression ---")
    model = get_model('logistic', seq_len, n_channels, n_classes)
    model.fit(X, y)
    
    # Test CleanLabelLinearAttack
    print("\n1. Clean Label Linear Attack:")
    attack = CleanLabelLinearAttack(model, eps_per_channel, target_class=0)
    attack.generate_trigger()
    X_poisoned, y_poisoned, poison_mask = attack.create_poisoned_dataset(X, y, poison_rate=0.1)
    print(f"   Trigger shape: {attack.trigger_pattern.shape}")
    print(f"   Poisoned samples: {poison_mask.sum()}")
    
    # Retrain on poisoned data
    model.fit(X_poisoned, y_poisoned)
    X_triggered, y_orig, source_mask = attack.create_triggered_test_set(X, y)
    asr, correct, total = calculate_attack_success_rate_sklearn(
        model, X_triggered, y_orig, 0, source_mask
    )
    print(f"   ASR: {asr:.2%}")
    
    # Test GradientBasedLinearAttack
    print("\n2. Gradient-Based Linear Attack:")
    attack = GradientBasedLinearAttack(model, eps_per_channel, target_class=0)
    X_poisoned, _, poison_mask = attack.create_poisoned_dataset(X, y, poison_rate=0.1)
    print(f"   Poisoned samples: {poison_mask.sum()}")
    
    # Test LabelFlipAttack
    print("\n3. Label Flip Attack:")
    attack = LabelFlipAttack(eps_per_channel, target_class=0)
    X_poisoned, y_poisoned, poison_mask = attack.create_poisoned_dataset(X, y, poison_rate=0.1)
    print(f"   Labels changed: {(y != y_poisoned).sum()}")
    
    # Test with SVM
    print("\n--- Testing with Linear SVM ---")
    model = get_model('linear_svm', seq_len, n_channels, n_classes)
    model.fit(X, y)
    
    attack = CleanLabelLinearAttack(model, eps_per_channel, target_class=0)
    attack.generate_trigger()
    print(f"   Trigger shape: {attack.trigger_pattern.shape}")
    
    print("\n" + "=" * 70)
    print("All attacks tested successfully!")
