"""
Linear Model Poisoning Attack

This module implements clean-label poisoning attacks specifically designed 
for linear classifiers on the UCI HAR dataset.

Key differences from LSTM attacks:
1. Works on flattened 561-dimensional feature vectors
2. Simpler feature space makes poisoning more effective
3. Uses gradient-based optimization without subspace constraints
4. Better for interpretability and analysis
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LinearPoisonAttack:
    """
    Clean-label poisoning attack for linear models.
    
    The attack works by:
    1. Selecting seed samples from the target's class
    2. Modifying them to have similar features to the target
    3. Keeping labels unchanged (clean-label)
    4. Training model learns association: modified pattern → target class
    5. When model sees target, it misclassifies
    """
    
    def __init__(self, model, eps=0.3, feature_layer=-4):
        """
        Args:
            model: Linear classifier model
            eps: Maximum perturbation magnitude (L-infinity)
            feature_layer: Which layer to extract features from (default: -4)
        """
        self.model = model
        self.eps = eps
        self.feature_layer = feature_layer
        self.device = DEVICE
        self.model.to(self.device)
        self.model.eval()
        
        self._features = None
        self._register_hook()
    
    def _register_hook(self):
        """Register forward hook to capture intermediate features."""
        def hook_fn(module, input, output):
            self._features = input[0] if isinstance(input, tuple) else input
        
        # Hook the specified layer
        layers = list(self.model.model.children())
        if self.feature_layer < len(layers):
            layers[self.feature_layer].register_forward_hook(hook_fn)
    
    def _get_features(self, x):
        """Extract features from the model."""
        self._features = None
        _ = self.model(x)
        return self._features
    
    def generate_poisons(self, seeds, target, steps=1000, lr=0.01, 
                        lambda_l2=0.01, verbose=True):
        """
        Generate poisoned samples from seeds.
        
        Args:
            seeds: Seed samples (numpy array, shape (P, 128, 9))
            target: Target sample (numpy array, shape (128, 9))
            steps: Optimization steps
            lr: Learning rate
            lambda_l2: L2 regularization weight
            verbose: Print progress
        
        Returns:
            Poisoned samples (numpy array, shape (P, 128, 9))
        """
        P = seeds.shape[0]
        
        # Flatten inputs for linear model
        seeds_flat = seeds.reshape(P, -1)  # (P, 1152)
        target_flat = target.reshape(1, -1)  # (1, 1152)
        
        # Convert to tensors
        seeds_tensor = torch.from_numpy(seeds_flat).float().to(self.device)
        target_tensor = torch.from_numpy(target_flat).float().to(self.device)
        
        # Get target features
        with torch.no_grad():
            feat_target = self._get_features(target_tensor).detach()
        
        # Initialize perturbations
        delta = torch.zeros_like(seeds_tensor, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=lr)
        
        # Optimization loop
        iterator = trange(steps, desc="Generating poisons") if verbose else range(steps)
        
        for it in iterator:
            optimizer.zero_grad()
            
            # Apply perturbation
            poisons = seeds_tensor + delta
            
            # Get features
            feat_poisons = self._get_features(poisons)
            
            # Feature collision loss (cosine similarity)
            feat_target_rep = feat_target.expand(P, -1)
            cos_sim = F.cosine_similarity(feat_poisons, feat_target_rep)
            loss_feat = 1 - cos_sim.mean()
            
            # Feature magnitude matching
            feat_p_norm = torch.norm(feat_poisons, dim=1)
            feat_t_norm = torch.norm(feat_target_rep, dim=1)
            loss_magnitude = F.mse_loss(feat_p_norm, feat_t_norm)
            
            # L2 regularization
            loss_l2 = lambda_l2 * torch.mean(delta ** 2)
            
            # Total loss
            loss = loss_feat + 0.1 * loss_magnitude + loss_l2
            
            # Backprop
            loss.backward()
            optimizer.step()
            
            # Clip perturbations to eps-ball
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            
            # Print progress
            if verbose and (it % 100 == 0 or it == steps - 1):
                if hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'feat': f'{loss_feat.item():.4f}',
                        'l2': f'{loss_l2.item():.4f}'
                    })
        
        # Return poisoned samples reshaped back to original format
        poisons_final = (seeds_tensor + delta).detach().cpu().numpy()
        return poisons_final.reshape(P, 128, 9)
    
    def evaluate_poison_quality(self, seeds, poisons, target):
        """
        Evaluate quality of generated poisons.
        
        Returns:
            dict with metrics
        """
        P = seeds.shape[0]
        
        # Flatten
        seeds_flat = seeds.reshape(P, -1)
        poisons_flat = poisons.reshape(P, -1)
        target_flat = target.reshape(1, -1)
        
        # Convert to tensors
        seeds_tensor = torch.from_numpy(seeds_flat).float().to(self.device)
        poisons_tensor = torch.from_numpy(poisons_flat).float().to(self.device)
        target_tensor = torch.from_numpy(target_flat).float().to(self.device)
        
        with torch.no_grad():
            # Get features
            feat_seeds = self._get_features(seeds_tensor)
            feat_poisons = self._get_features(poisons_tensor)
            feat_target = self._get_features(target_tensor)
            
            # Compute similarities
            sim_seeds_target = F.cosine_similarity(
                feat_seeds, feat_target.expand(P, -1)
            ).mean().item()
            
            sim_poisons_target = F.cosine_similarity(
                feat_poisons, feat_target.expand(P, -1)
            ).mean().item()
            
            # Perturbation magnitude
            pert_l2 = np.linalg.norm(poisons_flat - seeds_flat, axis=1).mean()
            pert_linf = np.abs(poisons_flat - seeds_flat).max()
        
        return {
            'similarity_seeds_target': sim_seeds_target,
            'similarity_poisons_target': sim_poisons_target,
            'improvement': sim_poisons_target - sim_seeds_target,
            'perturbation_l2': pert_l2,
            'perturbation_linf': pert_linf
        }


def optimize_linear_poisons(model, seeds, target, steps=1000, lr=0.01, 
                            eps=0.3, lambda_l2=0.01, verbose=True):
    """
    Convenience function for generating linear poisons.
    
    Args:
        model: Linear classifier
        seeds: Seed samples (P, 128, 9)
        target: Target sample (128, 9)
        steps: Optimization steps
        lr: Learning rate
        eps: Perturbation budget
        lambda_l2: L2 regularization
        verbose: Print progress
    
    Returns:
        poisons: Poisoned samples (P, 128, 9)
    """
    attack = LinearPoisonAttack(model, eps=eps)
    poisons = attack.generate_poisons(
        seeds, target, steps=steps, lr=lr, 
        lambda_l2=lambda_l2, verbose=verbose
    )
    
    if verbose:
        metrics = attack.evaluate_poison_quality(seeds, poisons, target)
        print(f"\nPoison Quality Metrics:")
        print(f"  Similarity improvement: {metrics['improvement']:.4f}")
        print(f"  Seeds→Target: {metrics['similarity_seeds_target']:.4f}")
        print(f"  Poisons→Target: {metrics['similarity_poisons_target']:.4f}")
        print(f"  Perturbation L2: {metrics['perturbation_l2']:.4f}")
        print(f"  Perturbation L∞: {metrics['perturbation_linf']:.4f}")
    
    return poisons
