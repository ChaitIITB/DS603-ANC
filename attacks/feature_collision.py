"""
Feature collision utilities for poisoning attacks.

Helper functions for analyzing and optimizing feature-space attacks.
"""

import numpy as np
import torch
import torch.nn.functional as F


def compute_feature_distance(model, samples1, samples2, metric='cosine'):
    """
    Compute distance between features of two sets of samples.
    
    Args:
        model: Trained model with feature extraction
        samples1: First set (numpy array)
        samples2: Second set (numpy array)
        metric: 'cosine' or 'euclidean'
    
    Returns:
        distances: Array of distances
    """
    device = next(model.parameters()).device
    
    # Flatten and convert
    s1_flat = samples1.reshape(samples1.shape[0], -1)
    s2_flat = samples2.reshape(samples2.shape[0], -1)
    
    s1_tensor = torch.from_numpy(s1_flat).float().to(device)
    s2_tensor = torch.from_numpy(s2_flat).float().to(device)
    
    with torch.no_grad():
        # Get features (assuming model has a method or we can hook)
        feat1 = model(s1_tensor)
        feat2 = model(s2_tensor)
        
        if metric == 'cosine':
            # Cosine similarity (1 = same, 0 = orthogonal, -1 = opposite)
            sim = F.cosine_similarity(feat1, feat2)
            return 1 - sim.cpu().numpy()  # Convert to distance
        
        elif metric == 'euclidean':
            dist = torch.norm(feat1 - feat2, dim=1)
            return dist.cpu().numpy()
        
        else:
            raise ValueError(f"Unknown metric: {metric}")


def project_to_subspace(vectors, basis):
    """
    Project vectors onto a subspace defined by basis vectors.
    
    Args:
        vectors: Input vectors (numpy array, shape (N, D))
        basis: Basis vectors (numpy array, shape (D, K))
    
    Returns:
        projected: Projected vectors (N, D)
        coefficients: Projection coefficients (N, K)
    """
    # Ensure basis is orthonormal
    basis_orth, _ = np.linalg.qr(basis)
    
    # Project: v_proj = U @ (U^T @ v)
    coefficients = vectors @ basis_orth  # (N, K)
    projected = coefficients @ basis_orth.T  # (N, D)
    
    return projected, coefficients


def compute_poison_effectiveness(model, poisons, target, labels_poison, label_target):
    """
    Measure how effective poisons are at causing misclassification.
    
    Args:
        model: Trained model
        poisons: Poisoned samples
        target: Target sample
        labels_poison: Labels of poison samples
        label_target: True label of target
    
    Returns:
        dict with effectiveness metrics
    """
    device = next(model.parameters()).device
    
    # Flatten inputs
    poisons_flat = poisons.reshape(poisons.shape[0], -1)
    target_flat = target.reshape(1, -1)
    
    poisons_tensor = torch.from_numpy(poisons_flat).float().to(device)
    target_tensor = torch.from_numpy(target_flat).float().to(device)
    
    model.eval()
    with torch.no_grad():
        # Get predictions
        poison_logits = model(poisons_tensor)
        target_logits = model(target_tensor)
        
        poison_preds = poison_logits.argmax(dim=1).cpu().numpy()
        target_pred = target_logits.argmax(dim=1).cpu().item()
        
        # Compute metrics
        poison_accuracy = np.mean(poison_preds == labels_poison)
        target_misclassified = (target_pred != label_target)
        
        # Confidence scores
        poison_conf = torch.softmax(poison_logits, dim=1).max(dim=1)[0].mean().item()
        target_conf = torch.softmax(target_logits, dim=1).max(dim=1)[0].item()
    
    return {
        'poison_accuracy': poison_accuracy,
        'target_misclassified': target_misclassified,
        'target_pred': target_pred,
        'poison_confidence': poison_conf,
        'target_confidence': target_conf
    }


def analyze_decision_boundary(model, samples, labels, num_classes=6):
    """
    Analyze how close samples are to decision boundaries.
    
    Args:
        model: Trained classifier
        samples: Input samples
        labels: True labels
        num_classes: Number of classes
    
    Returns:
        dict with boundary analysis
    """
    device = next(model.parameters()).device
    
    samples_flat = samples.reshape(samples.shape[0], -1)
    samples_tensor = torch.from_numpy(samples_flat).float().to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(samples_tensor)
        probs = torch.softmax(logits, dim=1)
        
        # Margin: difference between top-1 and top-2 probabilities
        top2_probs, top2_indices = torch.topk(probs, k=2, dim=1)
        margins = (top2_probs[:, 0] - top2_probs[:, 1]).cpu().numpy()
        
        # Confidence on true class
        labels_tensor = torch.from_numpy(labels).long().to(device)
        true_class_probs = probs.gather(1, labels_tensor.unsqueeze(1)).squeeze(1)
        true_class_conf = true_class_probs.cpu().numpy()
        
        # Prediction correctness
        preds = logits.argmax(dim=1).cpu().numpy()
        accuracy = np.mean(preds == labels)
    
    return {
        'margins_mean': margins.mean(),
        'margins_std': margins.std(),
        'true_class_confidence': true_class_conf.mean(),
        'accuracy': accuracy,
        'min_margin_idx': margins.argmin()  # Most vulnerable sample
    }
