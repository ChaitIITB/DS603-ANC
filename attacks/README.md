# Poisoning Attacks Module

This folder contains implementations of backdoor poisoning attacks for the UCI HAR dataset.

## Structure

```
attacks/
├── __init__.py              # Module exports
├── linear_poison.py         # Linear model poisoning attack
├── feature_collision.py     # Feature analysis utilities
└── README.md               # This file
```

## Linear Poisoning Attack

### Overview

The linear poisoning attack (`linear_poison.py`) implements clean-label backdoor attacks specifically designed for linear classifiers. This approach is:

- **Simpler**: Works on flattened 1152-dimensional vectors
- **More interpretable**: Linear decision boundaries are easier to analyze
- **More effective**: Linear models are often more vulnerable to poisoning
- **Faster**: No recurrent computations needed

### Key Components

#### `LinearPoisonAttack` Class

Main attack class that handles:
- Feature extraction via forward hooks
- Poison generation through gradient optimization
- Quality evaluation of generated poisons

**Parameters:**
- `model`: Linear classifier to attack
- `eps`: Maximum perturbation magnitude (L∞ constraint)
- `feature_layer`: Which layer to extract features from (default: -4)

**Key Methods:**
- `generate_poisons()`: Create poisoned samples from seeds
- `evaluate_poison_quality()`: Measure effectiveness of poisons

#### `optimize_linear_poisons()` Function

Convenience function for quick poison generation with automatic quality reporting.

### How It Works

1. **Seed Selection**: Choose samples from target's class (clean-label)
2. **Feature Collision**: Optimize seeds to have similar features to target
3. **Perturbation Constraint**: Clip perturbations within ε-ball
4. **Label Preservation**: Keep original labels (clean-label attack)
5. **Model Training**: Train on poisoned data
6. **Backdoor Trigger**: Target gets misclassified due to learned association

### Loss Function

```
L = L_feat + 0.1 * L_magnitude + λ * L_l2

where:
L_feat = 1 - cosine_similarity(features_poison, features_target)
L_magnitude = MSE(||features_poison||, ||features_target||)
L_l2 = mean(delta²)
```

## Feature Collision Utilities

The `feature_collision.py` module provides analysis tools:

### Functions

- **`compute_feature_distance()`**: Measure distance between feature representations
- **`project_to_subspace()`**: Project vectors onto learned subspace
- **`compute_poison_effectiveness()`**: Evaluate attack success metrics
- **`analyze_decision_boundary()`**: Analyze proximity to decision boundaries

## Usage

### Basic Usage

```python
from attacks.linear_poison import LinearPoisonAttack
from models.linear_model import LinearModel

# Load model and data
model = LinearModel(input_size=1152, num_classes=6)
model.load_state_dict(torch.load("model.pth"))

# Create attack
attack = LinearPoisonAttack(model, eps=0.5)

# Generate poisons
seeds = X_train[seed_indices]  # Shape: (P, 128, 9)
target = X_train[target_idx]    # Shape: (128, 9)

poisons = attack.generate_poisons(
    seeds=seeds,
    target=target,
    steps=1000,
    lr=0.02,
    lambda_l2=0.005
)
```

### Complete Attack Pipeline

See `test_linear_attack.py` for a complete example that:
1. Trains a clean linear model
2. Generates poisoned samples
3. Trains a poisoned model
4. Evaluates attack success

Run with:
```bash
python test_linear_attack.py
```

## Configuration

### Recommended Hyperparameters

**For Linear Models:**
- `eps`: 0.3 - 0.5 (perturbation budget)
- `steps`: 1000 - 1500 (optimization iterations)
- `lr`: 0.01 - 0.02 (learning rate)
- `lambda_l2`: 0.005 - 0.01 (regularization)
- `num_poisons`: 200 - 400 (number of poisoned samples)

**For Training Poisoned Model:**
- `epochs`: 50 - 70
- `batch_size`: 256
- `lr`: 1e-3 to 5e-4
- Add 30-50 target replicas

## Attack Success Metrics

The attack is evaluated on:

1. **Attack Success Rate**: Does target get misclassified to desired class?
2. **Stealthiness**: Model accuracy drop < 5%
3. **Perturbation Size**: L2 norm < 2.0
4. **Feature Similarity**: Cosine similarity > 0.8

## Advantages Over LSTM Attacks

1. **Faster Training**: Linear models train 10x faster than LSTMs
2. **Better Interpretability**: Can analyze decision boundaries directly
3. **Higher Success Rate**: Linear models more vulnerable to feature collision
4. **Easier Debugging**: Simpler architecture makes issues easier to identify
5. **Lower Computational Cost**: No GPU required for reasonable performance

## Comparison with LSTM Attacks

| Aspect | Linear Attack | LSTM Attack |
|--------|---------------|-------------|
| Input Size | 1152 (flattened) | 128×9 (sequential) |
| Training Time | ~2 min | ~15 min |
| Poison Generation | ~30 sec | ~5 min |
| Success Rate | 70-85% | 40-60% |
| Memory Usage | Low | High |
| Interpretability | High | Low |

## Troubleshooting

### Attack Not Working?

1. **Increase perturbation budget**: Try `eps=0.7` or higher
2. **More poisons**: Use 400-500 poisoned samples
3. **More target replicas**: Add 50-100 copies of target
4. **Longer optimization**: Increase steps to 1500-2000
5. **Lower regularization**: Reduce `lambda_l2` to 0.003

### Model Accuracy Dropping Too Much?

1. **Reduce perturbation**: Lower `eps` to 0.3
2. **Fewer poisons**: Use 100-200 samples
3. **Stronger regularization**: Increase `lambda_l2` to 0.02
4. **Better seeds**: Choose seeds more similar to target

## References

This implementation is based on clean-label poisoning attacks:
- Feature collision optimization
- Gradient-based perturbation crafting
- L∞ constraint for imperceptibility

## Future Enhancements

- [ ] Add support for multiple targets simultaneously
- [ ] Implement adaptive perturbation budgets per sample
- [ ] Add defense mechanisms (spectral signatures, activation clustering)
- [ ] Support for transfer attacks (different surrogate/victim models)
- [ ] Visualization of decision boundaries before/after poisoning
