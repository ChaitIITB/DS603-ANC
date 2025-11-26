# Poisoning Attacks

Clean-label backdoor attacks for UCI HAR dataset.

## Files

- `simple_gradient_attack.py` - Simple gradient-based attack (recommended)
- `linear_poison.py` - Feature collision attack
- `feature_collision.py` - Analysis utilities

## Quick Start

```bash
python test_simple_linear.py
```

## Simple Gradient Attack (Recommended)

**Strategy**: Make poisoned samples output target class, train model to misclassify target.

**Usage**:
```python
from attacks.simple_gradient_attack import simple_linear_poison

X_poisoned, y_poisoned, info = simple_linear_poison(
    X_train, y_train, 
    target_idx=10,
    num_poisons=300,
    steps=500,
    lr=0.05,
    eps=0.6
)
```

**Key Parameters**:
- `num_poisons`: 200-500 samples
- `eps`: 0.5-0.8 perturbation budget
- `steps`: 500-1000 optimization iterations
- `lr`: 0.03-0.1 learning rate

## Linear Poison Attack

**Strategy**: Feature collision using forward hooks.

**Usage**:
```python
from attacks.linear_poison import LinearPoisonAttack

attack = LinearPoisonAttack(model, eps=0.5)
poisons = attack.generate_poisons(seeds, target, steps=1000)
```

## Troubleshooting

**Attack not working?**
- Increase `num_poisons` to 400-500
- Increase `eps` to 0.8-1.0
- Use 1000+ optimization steps
- Train poisoned model 50+ epochs

**Accuracy dropping too much?**
- Reduce `num_poisons` to 150-200
- Lower `eps` to 0.3-0.4
- Increase regularization in training

## Performance

| Method | Success Rate | Training Time | GPU Memory |
|--------|--------------|---------------|------------|
| Simple Gradient | 70-85% | ~2 min | Low |
| Feature Collision | 60-75% | ~3 min | Medium |

## Advantages vs LSTM Attacks

- 10x faster training
- Simpler implementation
- Higher success rate
- Better interpretability
- No subspace constraints needed

