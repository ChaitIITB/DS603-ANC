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
    
    def get_features(self, x):
        x = x.transpose(1, 2)
        return self.net(x)       # (N, 64)
    

class FreqPGDCollisionAttack:
    """
    Clean-label backdoor attack combining:
      1. Surrogate model trained on FFT magnitudes (= FreqDomainAttack surrogate)
      2. PGD feature-collision to find an OPTIMIZED frequency trigger
         (instead of the fixed magnitude-boost trigger)
      3. Apply that optimized trigger to target-class training samples
      4. Train target model on poisoned data → backdoor embedded

    At test time: stamp the same PGD-optimized trigger → misclassified as target.
    """

    def __init__(
        self,
        eps_per_channel,
        target_class: int = 0,
        trigger_strength: float = 0.2,
        top_percent: float = 10,
        device: str = "cuda",
    ):
        self.eps_per_channel  = torch.tensor(
            eps_per_channel, dtype=torch.float32, device=device
        )
        self.target_class     = target_class
        self.trigger_strength = trigger_strength
        self.top_percent      = top_percent
        self.device           = device

        self.surrogate        = None   # SmallFreqCNN
        self.mask             = None   # important freq-bin mask
        self.trigger_delta    = None   # optimized frequency perturbation (freq_bins, n_ch)

    # ------------------------------------------------------------------
    # Step 1: Train surrogate + build importance mask
    # ------------------------------------------------------------------

    def _fit_surrogate(self, X: np.ndarray, y: np.ndarray,
                       epochs: int = 20, batch_size: int = 256):
        """
        Train SmallFreqCNN on FFT magnitudes, then compute gradient-based
        importance mask — identical to FreqDomainAttack._fit_importance.
        """
        print("[FreqPGDCollision] Training surrogate on FFT magnitudes...")

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long,    device=self.device)

        # FFT magnitude: (N, freq_bins, n_channels)
        X_f = torch.abs(torch.fft.rfft(X_t, dim=1))

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_f, y_t),
            batch_size=batch_size, shuffle=True,
        )

        n_channels = X_f.shape[2]
        n_classes  = len(torch.unique(y_t))
        self.surrogate = SmallFreqCNN(n_channels, n_classes).to(self.device)
        opt = torch.optim.Adam(self.surrogate.parameters(), lr=1e-3)

        self.surrogate.train()
        for epoch in range(epochs):
            for xb, yb in loader:
                logits = self.surrogate(xb)
                loss   = F.cross_entropy(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs} — loss: {loss.item():.4f}")

        # --- gradient-based importance (same as FreqDomainAttack) ---
        print("[FreqPGDCollision] Computing importance mask...")
        X_f_grad = X_f.detach().requires_grad_(True)
        self.surrogate.eval()
        loss = F.cross_entropy(self.surrogate(X_f_grad), y_t)
        loss.backward()

        importance = X_f_grad.grad.abs().mean(dim=0)   # (freq_bins, n_ch)
        importance = importance / (importance.max() + 1e-8)

        threshold  = torch.quantile(importance.flatten(), 1.0 - self.top_percent / 100.0)
        self.mask  = (importance >= threshold).detach()

        print(f"  Mask covers {self.mask.sum().item()} / {self.mask.numel()} bins.")

    # ------------------------------------------------------------------
    # Step 2: PGD feature-collision to find universal trigger
    # ------------------------------------------------------------------

    def _optimize_trigger(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_steps: int = 500,
        step_size: float = 0.01,
        n_anchor_samples: int = 64,
        batch_size: int = 64,
    ):
        """
        Find a universal frequency-domain trigger δ_freq such that:

            φ_surrogate(iRFFT(mag_base * (1 + δ))) ≈ φ_surrogate(target_anchor)

        averaged over many base samples from non-target classes.

        δ_freq is constrained so the time-domain perturbation
        stays within ±eps_per_channel.

        Only the masked (important) frequency bins are optimised;
        the rest are held at zero.
        """
        print("[FreqPGDCollision] Optimizing universal frequency trigger via PGD...")

        self.surrogate.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long,    device=self.device)

        # --- target anchor: mean surrogate features of target-class samples ---
        target_idx = np.where(y == self.target_class)[0]
        anchor_idx = np.random.choice(
            target_idx,
            min(n_anchor_samples, len(target_idx)),
            replace=False,
        )
        X_target = X_t[anchor_idx]                             # (K, seq_len, n_ch)
        X_target_f = torch.abs(torch.fft.rfft(X_target, dim=1))

        with torch.no_grad():
            phi_target = self.surrogate.get_features(X_target_f).mean(dim=0, keepdim=True)
            # (1, feat_dim)

        # --- source samples: non-target class ---
        source_idx = np.where(y != self.target_class)[0]
        chosen_src = np.random.choice(
            source_idx,
            min(n_anchor_samples, len(source_idx)),
            replace=False,
        )
        X_source = X_t[chosen_src]                             # (M, seq_len, n_ch)

        seq_len  = X_source.shape[1]

        # FFT of source samples
        X_src_f   = torch.fft.rfft(X_source, dim=1)           # (M, freq_bins, n_ch)
        X_src_mag = torch.abs(X_src_f).detach()
        X_src_mag_safe = X_src_mag.clamp(min=1e-6)
        X_src_phase = torch.angle(X_src_f).detach()

        # --- universal δ (freq_bins, n_ch), only non-zero on mask ---
        # Initialise with small magnitude-boost on masked bins
        delta = torch.where(
            self.mask,
            torch.full_like(X_src_mag[0], self.trigger_strength),
            torch.zeros_like(X_src_mag[0]),
        ).detach().requires_grad_(True)                        # (freq_bins, n_ch)

        lr = step_size
        best_loss   = float('inf')
        best_delta  = delta.detach().clone()

        for step in range(n_steps):
            # Process source samples in mini-batches to save memory
            total_loss = torch.tensor(0.0, device=self.device)
            n_mini = 0

            for mb_start in range(0, len(X_source), batch_size):
                mb_mag   = X_src_mag_safe[mb_start:mb_start + batch_size]   # (b, freq_bins, n_ch)
                mb_phase = X_src_phase[mb_start:mb_start + batch_size]

                # Apply universal delta (broadcast over batch)
                mag_perturbed = mb_mag * (1.0 + delta.unsqueeze(0))         # (b, freq_bins, n_ch)

                # Clamp to keep mag positive
                mag_perturbed = mag_perturbed.clamp(min=0.0)

                # Reconstruct time-domain
                X_f_new = mag_perturbed * torch.exp(1j * mb_phase)
                x_recon = torch.fft.irfft(X_f_new, n=seq_len, dim=1)       # (b, seq_len, n_ch)

                if torch.isnan(x_recon).any():
                    continue

                phi_p = self.surrogate.get_features(
                    torch.abs(torch.fft.rfft(x_recon, dim=1))
                )                                                            # (b, feat_dim)

                # Collision loss: MSE to target anchor features
                loss_mb = F.mse_loss(
                    phi_p,
                    phi_target.expand(phi_p.shape[0], -1),
                )

                total_loss = total_loss + loss_mb
                n_mini += 1

            if n_mini == 0:
                continue

            total_loss = total_loss / n_mini

            if delta.grad is not None:
                delta.grad.zero_()

            total_loss.backward()

            if delta.grad is None or torch.isnan(delta.grad).any():
                lr *= 0.5
                delta = delta.detach().requires_grad_(True)
                continue

            if step % 100 == 0:
                print(f"  step {step:4d} | collision_loss={total_loss.item():.6f}")

            # Track best
            if total_loss.item() < best_loss:
                best_loss  = total_loss.item()
                best_delta = delta.detach().clone()

            with torch.no_grad():
                # Sign-gradient step, masked (don't touch unimportant bins)
                grad_sign = delta.grad.sign()
                grad_sign = torch.where(self.mask, grad_sign, torch.zeros_like(grad_sign))
                delta -= lr * grad_sign

                # ℓ∞ projection: enforce time-domain budget per channel
                # Approximate: reconstruct with mean source mag and check
                mean_mag   = X_src_mag_safe.mean(dim=0)                    # (freq_bins, n_ch)
                mag_proj   = mean_mag * (1.0 + delta)
                X_f_proj   = mag_proj * torch.exp(1j * X_src_phase.mean(dim=0))
                x_proj     = torch.fft.irfft(X_f_proj, n=seq_len, dim=0)  # (seq_len, n_ch) - mean

                # Per-channel max perturbation
                mean_src   = torch.fft.irfft(
                    mean_mag * torch.exp(1j * X_src_phase.mean(dim=0)),
                    n=seq_len, dim=0,
                )
                pert       = x_proj - mean_src
                excess     = pert.abs() - self.eps_per_channel.unsqueeze(0)  # (seq_len, n_ch)
                overshoot  = excess.clamp(min=0).max()

                if overshoot > 0:
                    # Scale delta down proportionally
                    delta = delta * (self.eps_per_channel.unsqueeze(0).max() /
                                     (self.eps_per_channel.unsqueeze(0).max() + overshoot))

                # Keep delta bounded: mag*(1+delta) must stay positive
                delta = delta.clamp(min=-0.99, max=10.0)

                # Zero out unmasked bins
                delta = torch.where(self.mask, delta, torch.zeros_like(delta))

            delta = delta.detach().requires_grad_(True)

        print(f"  Best collision loss: {best_loss:.6f}")
        self.trigger_delta = best_delta.detach()

    # ------------------------------------------------------------------
    # Apply the optimized trigger to a sample
    # ------------------------------------------------------------------

    def apply_trigger(self, x: np.ndarray) -> np.ndarray:
        """
        Stamp the PGD-optimized frequency trigger onto a single sample.
        x: (seq_len, n_channels)
        """
        x_t     = torch.tensor(x, dtype=torch.float32, device=self.device)
        seq_len = x_t.shape[0]

        X_f   = torch.fft.rfft(x_t, dim=0)                    # (freq_bins, n_ch)
        mag   = torch.abs(X_f).clamp(min=1e-6)
        phase = torch.angle(X_f)

        # Apply learned delta
        mag_triggered = mag * (1.0 + self.trigger_delta)
        mag_triggered = mag_triggered.clamp(min=0.0)
        mag_triggered = torch.nan_to_num(mag_triggered, nan=0.0, posinf=0.0)

        X_f_new     = mag_triggered * torch.exp(1j * phase)
        x_triggered = torch.fft.irfft(X_f_new, n=seq_len, dim=0)

        perturbation = torch.clamp(
            x_triggered - x_t,
            -self.eps_per_channel,
             self.eps_per_channel,
        )
        result = x_t + perturbation

        if torch.isnan(result).any() or torch.isinf(result).any():
            return x

        return result.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Create poisoned dataset (= FreqDomainAttack.create_poisoned_dataset)
    # ------------------------------------------------------------------

    def create_poisoned_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        poison_rate: float = 0.1,
        target_samples_only: bool = True,
        surrogate_epochs: int = 20,
        n_trigger_steps: int = 500,
        trigger_step_size: float = 0.01,
    ):
        """
        Full pipeline:
          1. Train surrogate + build mask  (once)
          2. Optimize universal trigger    (once)
          3. Stamp trigger on selected samples (same as FreqDomainAttack)
        """
        # Step 1 + 2: fit surrogate and optimise trigger if not done yet
        if self.surrogate is None:
            self._fit_surrogate(X, y, epochs=surrogate_epochs)

        if self.trigger_delta is None:
            self._optimize_trigger(
                X, y,
                n_steps=n_trigger_steps,
                step_size=trigger_step_size,
            )

        X_poisoned  = X.copy()
        y_poisoned  = y.copy()
        poison_mask = np.zeros(len(X), dtype=bool)

        # Same selection logic as FreqDomainAttack
        if target_samples_only:
            candidate_indices = np.where(y == self.target_class)[0]
        else:
            candidate_indices = np.arange(len(X))

        n_poison       = max(1, int(len(candidate_indices) * poison_rate))
        poison_indices = np.random.choice(candidate_indices, n_poison, replace=False)

        print(f"[FreqPGDCollision] Stamping trigger on {n_poison} samples "
              f"(rate={poison_rate}, target_class={self.target_class}, "
              f"target_only={target_samples_only})")

        for idx in tqdm(poison_indices, desc="Poisoning"):
            triggered = self.apply_trigger(X[idx])
            if not np.isnan(triggered).any():
                X_poisoned[idx] = triggered
                poison_mask[idx] = True
            else:
                print(f"  WARNING: nan for idx={idx} — keeping clean")

        print(f"[FreqPGDCollision] Done. {poison_mask.sum()} / {len(X)} poisoned.")
        return X_poisoned, y_poisoned, poison_mask

    # ------------------------------------------------------------------
    # Test-time triggered set (identical API to FreqDomainAttack)
    # ------------------------------------------------------------------

    def create_triggered_test_set(
        self,
        X: np.ndarray,
        y: np.ndarray,
        source_classes=None,
    ):
        if source_classes is None:
            source_classes = [c for c in np.unique(y) if c != self.target_class]

        source_mask = np.isin(y, source_classes)
        X_triggered = X.copy()

        for idx in tqdm(np.where(source_mask)[0], desc="Triggering test set"):
            X_triggered[idx] = self.apply_trigger(X[idx])

        print(f"[FreqPGDCollision] Triggered test set: "
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
