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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
# Small surrogate model
# --------------------------------------------------
class SmallFreqCNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 1024, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(1024, 512, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, F)
        x = self.net(x)
        x = x.squeeze(-1)
        return self.fc(x)
    
class WhiteBoxFreqFeatureCollisionAttack:
    """
    White-box clean-label backdoor attack via feature collision in frequency domain.
    Uses the model's own get_features() method — no hooks needed.
    """

    def __init__(
        self,
        model: nn.Module,
        eps_per_channel,
        target_class: int = 0,
        trigger_strength: float = 0.2,
        top_percent: float = 10,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.eps_per_channel = torch.tensor(
            eps_per_channel, dtype=torch.float32, device=device
        )
        self.target_class = target_class
        self.trigger_strength = trigger_strength
        self.top_percent = top_percent
        self.device = device

        self.importance_matrix = None
        self.mask = None

    # ------------------------------------------------------------------
    # Feature extraction via get_features()
    # ------------------------------------------------------------------

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, seq_len, n_channels) — calls model.get_features() directly.
        No hooks, no layer name strings.
        """
        self.model.eval()
        return self.model.get_features(x)

    # ------------------------------------------------------------------
    # Importance: gradient of feature norm w.r.t. frequency magnitudes
    # ------------------------------------------------------------------

    def _fit_importance(self, X: np.ndarray, y: np.ndarray):
        """
        One forward+backward through the TARGET model in frequency space.

        We reconstruct a zero-phase time-domain signal from FFT magnitudes,
        push it through get_features(), and take gradients w.r.t. magnitudes
        to identify which frequency bins matter most.
        """
        print("[WhiteBoxFreqCollision] Computing frequency importance via get_features()...")

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long,    device=self.device)

        # FFT magnitude: (N, freq_bins, n_channels)
        X_f   = torch.fft.rfft(X_t, dim=1)
        X_mag = torch.abs(X_f)
        X_mag.requires_grad_(True)

        self.model.eval()

        # Zero-phase reconstruction so the model sees its expected input format
        X_recon = torch.fft.irfft(X_mag, n=X_t.shape[1], dim=1)  # (N, seq_len, n_ch)

        # Use get_features() + cross-entropy on a linear probe over feature norm
        # (avoids needing the classification head to be differentiable w.r.t. X_mag)
        features = self._get_features(X_recon)                    # (N, feat_dim)
        logits   = self.model(X_recon)
        loss     = nn.functional.cross_entropy(logits, y_t)
        loss.backward()

        # Gradient magnitude averaged over samples → (freq_bins, n_channels)
        grads      = X_mag.grad
        importance = grads.abs().mean(dim=0)
        importance = importance / (importance.max() + 1e-8)

        self.importance_matrix = importance.detach()

        threshold  = torch.quantile(importance.flatten(), 1.0 - self.top_percent / 100.0)
        self.mask  = importance >= threshold

        print(f"  Done. Mask covers {self.mask.sum().item()} / "
              f"{self.mask.numel()} freq-channel bins.")

    # ------------------------------------------------------------------
    # Fixed trigger (test-time)
    # ------------------------------------------------------------------

    def apply_trigger(self, x: np.ndarray) -> np.ndarray:
        """
        Boost important frequency-bin magnitudes in a single sample.
        x: (seq_len, n_channels)
        """
        x_t   = torch.tensor(x, dtype=torch.float32, device=self.device)
        X_f   = torch.fft.rfft(x_t, dim=0)
        mag   = torch.abs(X_f)
        phase = torch.angle(X_f)

        mag_triggered = torch.where(
            self.mask,
            mag * (1.0 + self.trigger_strength),
            mag,
        )

        X_f_new     = mag_triggered * torch.exp(1j * phase)
        x_triggered = torch.fft.irfft(X_f_new, n=x_t.shape[0], dim=0)

        perturbation = torch.clamp(
            x_triggered - x_t,
            -self.eps_per_channel,
             self.eps_per_channel,
        )
        return (x_t + perturbation).detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Core: per-sample feature collision optimisation
    # ------------------------------------------------------------------

    def compute_poison_sample(
        self,
        base_sample: np.ndarray,
        target_sample: np.ndarray,
        n_steps: int = 500,
        step_size: float = 0.01,
        decay: float = 1.0,
        feature_loss_weight: float = 1.0,
        input_reg_weight: float = 0.05,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Optimise δ_freq so that get_features(iRFFT(mag*(1+δ))) ≈ get_features(target).

        Constraint: ||iRFFT(δ_freq) - base||_∞ ≤ eps_per_channel (per channel).
        """
        self.model.eval()
        seq_len = base_sample.shape[0]

        # Target features — fixed, no grad
        t_tensor = (
            torch.tensor(target_sample, dtype=torch.float32)
            .unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            phi_target = self._get_features(t_tensor).detach()

        # Base in frequency space
        base_t     = torch.tensor(base_sample, dtype=torch.float32, device=self.device)
        base_freq  = torch.fft.rfft(base_t, dim=0)
        base_mag   = torch.abs(base_freq)
        base_phase = torch.angle(base_freq)          # held constant throughout

        # Initialise δ with the trigger boost on important bins
        delta_init = torch.where(
            self.mask,
            torch.full_like(base_mag, self.trigger_strength),
            torch.zeros_like(base_mag),
        )
        delta = delta_init.clone().requires_grad_(True)

        base_input = base_t.unsqueeze(0)
        lr = step_size

        for step in range(n_steps):
            if delta.grad is not None:
                delta.grad.zero_()

            # Reconstruct time-domain from perturbed magnitude
            mag_perturbed = base_mag * (1.0 + delta)
            X_f_new       = mag_perturbed * torch.exp(1j * base_phase)
            x_recon       = torch.fft.irfft(X_f_new, n=seq_len, dim=0)   # (seq_len, n_ch)
            x_recon_batch = x_recon.unsqueeze(0)                          # (1, seq_len, n_ch)

            phi_p = self._get_features(x_recon_batch)

            loss_feat = feature_loss_weight * torch.mean((phi_p - phi_target) ** 2)
            loss_reg  = input_reg_weight    * torch.mean((x_recon_batch - base_input) ** 2)
            loss      = loss_feat + loss_reg
            loss.backward()

            if verbose and step % 100 == 0:
                print(f"    step {step:4d} | feat={loss_feat.item():.6f} "
                      f"| reg={loss_reg.item():.6f}")

            with torch.no_grad():
                delta -= lr * delta.grad.sign()

                # Project: enforce ℓ∞ budget in time domain, back-project to δ
                mag_proj  = base_mag * (1.0 + delta)
                X_f_proj  = mag_proj * torch.exp(1j * base_phase)
                x_proj    = torch.fft.irfft(X_f_proj, n=seq_len, dim=0)

                excess    = (x_proj - base_t) - torch.clamp(
                    x_proj - base_t,
                    -self.eps_per_channel,
                     self.eps_per_channel,
                )
                excess_f  = torch.fft.rfft(excess, dim=0)
                delta     = delta - (torch.abs(excess_f) / (base_mag + 1e-8)).detach()

            delta = delta.detach().requires_grad_(True)
            lr   *= decay

        # Final reconstruction
        with torch.no_grad():
            mag_final = base_mag * (1.0 + delta)
            X_f_final = mag_final * torch.exp(1j * base_phase)
            x_final   = torch.fft.irfft(X_f_final, n=seq_len, dim=0)

            perturbation = torch.clamp(
                x_final - base_t,
                -self.eps_per_channel,
                 self.eps_per_channel,
            )
            poison = (base_t + perturbation).cpu().numpy()

        return poison

    # ------------------------------------------------------------------
    # Dataset-level poisoning
    # ------------------------------------------------------------------

    def create_poisoned_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        poison_rate: float = 0.1,
        target_samples_only: bool = False,
        n_steps: int = 500,
        step_size: float = 0.01,
        input_reg_weight: float = 0.05,
        verbose: bool = False,
    ):
        if self.mask is None:
            self._fit_importance(X, y)

        X_poisoned  = X.copy()
        y_poisoned  = y.copy()
        poison_mask = np.zeros(len(X), dtype=bool)

        # For feature collision: poison NON-target samples (clean label kept)
        source_idx = (np.where(y == self.target_class)[0]
                      if target_samples_only
                      else np.where(y != self.target_class)[0])
        target_idx = np.where(y == self.target_class)[0]

        n_poison = max(1, int(len(source_idx) * poison_rate))
        chosen   = np.random.choice(source_idx, n_poison, replace=False)

        print(f"\n[WhiteBoxFreqCollision] Poisoning {n_poison} samples "
              f"(rate={poison_rate}, target_class={self.target_class})")

        for i, idx in enumerate(chosen):
            anchor_idx = np.random.choice(target_idx)
            if verbose:
                print(f"  [{i+1}/{n_poison}] src_label={y[idx]} anchor={anchor_idx}")

            X_poisoned[idx] = self.compute_poison_sample(
                base_sample      = X[idx],
                target_sample    = X[anchor_idx],
                n_steps          = n_steps,
                step_size        = step_size,
                input_reg_weight = input_reg_weight,
                verbose          = verbose,
            )
            poison_mask[idx] = True

        print(f"[WhiteBoxFreqCollision] Done. {poison_mask.sum()} / {len(X)} poisoned.\n")
        return X_poisoned, y_poisoned, poison_mask

    # ------------------------------------------------------------------
    # Test-time trigger
    # ------------------------------------------------------------------

    def create_triggered_test_set(self, X, y, source_classes=None):
        if source_classes is None:
            source_classes = [c for c in np.unique(y) if c != self.target_class]

        source_mask = np.isin(y, source_classes)
        X_triggered = X.copy()
        for idx in np.where(source_mask)[0]:
            X_triggered[idx] = self.apply_trigger(X[idx])

        print(f"[WhiteBoxFreqCollision] Triggered test set: "
              f"{source_mask.sum()} samples from {source_classes}")
        return X_triggered, y, source_mask
    
# --------------------------------------------------
# Main Attack Class (API-compatible)
# --------------------------------------------------
class FreqDomainAttack:
    def __init__(self, model, eps_per_channel,
                 target_class=0, trigger_strength=0.2,
                 device='cuda', top_percent=10):

        self.model = model  # kept for API consistency (unused)
        self.eps_per_channel = torch.tensor(eps_per_channel, device=device)
        self.target_class = target_class
        self.trigger_strength = trigger_strength
        self.device = device
        self.top_percent = top_percent

        self.surrogate = None
        self.importance_matrix = None
        self.mask = None

    # --------------------------------------------------
    # Train surrogate + compute importance
    # --------------------------------------------------
    def _fit_importance(self, X, y, epochs=5, batch_size=256):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.long, device=self.device)

        # FFT → magnitude
        X_f = torch.fft.rfft(X, dim=1)
        X_f = torch.abs(X_f)

        dataset = torch.utils.data.TensorDataset(X_f, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        n_channels = X_f.shape[2]
        n_classes = len(torch.unique(y))

        self.surrogate = SmallFreqCNN(n_channels, n_classes).to(self.device)
        optimizer = torch.optim.Adam(self.surrogate.parameters(), lr=1e-3)

        # Train
        self.surrogate.train()
        for i in range(epochs):
            for xb, yb in loader:
                logits = self.surrogate(xb)
                loss = F.cross_entropy(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'{loss} at {i+1}th epoch')

        # -----------------------------
        # Gradient-based importance
        # -----------------------------
        X_f.requires_grad_(True)
        self.surrogate.eval()

        logits = self.surrogate(X_f)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        grads = X_f.grad
        importance = grads.abs().mean(dim=0)

        importance = importance / (importance.max() + 1e-8)
        self.importance_matrix = importance.detach()

        # Build mask
        threshold = torch.quantile(
            importance.flatten(),
            1 - self.top_percent / 100.0
        )
        self.mask = importance >= threshold

    # --------------------------------------------------
    # Apply trigger (frequency domain)
    # --------------------------------------------------
    def apply_trigger(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=self.device)

        X_f = torch.fft.rfft(x, dim=0)

        mag = torch.abs(X_f)
        phase = torch.angle(X_f)

        mag = torch.where(
            self.mask,
            mag * (1 + self.trigger_strength),
            mag
        )

        X_f_new = mag * torch.exp(1j * phase)
        x_triggered = torch.fft.irfft(X_f_new, dim=0)

        perturbation = torch.clamp(
            x_triggered - x,
            -self.eps_per_channel,
            self.eps_per_channel
        )

        return (x + perturbation).detach().cpu().numpy()

    # --------------------------------------------------
    # Poison dataset (same API)
    # --------------------------------------------------
    def create_poisoned_dataset(self, X, y, poison_rate=0.1,
                               target_samples_only=True):

        # Compute importance ONCE
        if self.mask is None:
            print("Computing frequency-domain importance...")
            self._fit_importance(X, y, epochs=160)

        X_poisoned = X.copy()
        y_poisoned = y.copy()
        poison_mask = np.zeros(len(X), dtype=bool)

        if target_samples_only:
            target_indices = np.where(y == self.target_class)[0]
        else:
            target_indices = np.arange(len(X))

        n_poison = max(1, int(len(target_indices) * poison_rate))
        poison_indices = np.random.choice(target_indices, n_poison, replace=False)

        print(f"Creating poisoned dataset (FreqDomainAttack):")
        print(f"  Target class: {self.target_class}")
        print(f"  Poisoned samples: {n_poison}")

        for idx in poison_indices:
            X_poisoned[idx] = self.apply_trigger(X[idx])
            poison_mask[idx] = True

        return X_poisoned, y_poisoned, poison_mask

    # --------------------------------------------------
    # Triggered test set (same API)
    # --------------------------------------------------
    def create_triggered_test_set(self, X, y, source_classes=None):

        if source_classes is None:
            source_classes = [c for c in np.unique(y) if c != self.target_class]

        source_mask = np.isin(y, source_classes)
        source_indices = np.where(source_mask)[0]

        X_triggered = X.copy()

        for idx in source_indices:
            X_triggered[idx] = self.apply_trigger(X[idx])

        print(f"Created triggered test set (FreqDomainAttack):")
        print(f"  Triggered samples: {len(source_indices)}")

        return X_triggered, y, source_mask

# does not particularly work well for black box attacks 
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
        self.model.train()
        
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
            # features = features.detach()
            # features.requires_grad=True
            
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
