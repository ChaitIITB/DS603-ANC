# WISDM Activity Recognition - Backdoor Attack

Complete implementation of backdoor poison attacks on the WISDM Activity Recognition dataset.

## Dataset Overview

**WISDM (Wireless Sensor Data Mining) Activity Recognition Dataset v1.1**
- **Activities**: 6 classes (Walking, Jogging, Upstairs, Downstairs, Sitting, Standing)
- **Sensors**: 3-axis accelerometer (x, y, z)
- **Sampling Rate**: 20 Hz (1 sample every 50ms)
- **Users**: 36 participants
- **Total Samples**: ~1.1 million accelerometer readings
- **Window Size**: 80 samples (4 seconds at 20Hz)

## Project Structure

```
project/
├── data/
│   ├── WISDM_ar_latest/
│   │   └── WISDM_ar_v1.1/
│   │       └── WISDM_ar_v1.1_raw.txt    # Raw dataset
│   ├── load_wisdm_data.py               # Data preprocessing
│   └── wisdm_processed/                 # Processed data (generated)
│       ├── X_train.npy, X_test.npy
│       ├── y_train.npy, y_test.npy
│       └── eps_per_channel.npy
├── models/
│   ├── wisdm_models.py                  # LSTM models for WISDM
│   └── wisdm_surrogate.pth              # Trained surrogate (generated)
├── wisdm_subspace/                      # Subspace matrices (generated)
│   ├── U.npy, M.npy, mu_global.npy
│   └── metadata.json
├── train_wisdm_surrogate.py             # Train surrogate model
├── compute_wisdm_subspace.py            # Compute PCA subspace
├── wisdm_poison_optimize.py             # Poison optimization
└── wisdm_attack_main.py                 # Main attack pipeline
```

## Attack Overview

This implementation demonstrates a **clean-label backdoor attack** on activity recognition:

1. **Target Sample**: Select a sample from the training set (e.g., "Walking" activity)
2. **Attack Goal**: Misclassify the target as a different activity (e.g., "Jogging")
3. **Method**: Poison training samples from the target's class using feature collision
4. **Constraint**: Keep perturbations small and within subspace to remain stealthy

### Key Features

- **Clean-label attack**: Poison samples keep their original labels
- **Feature collision**: Optimize poisons to have similar features as target
- **Subspace projection**: Constrain perturbations to low-dimensional PCA subspace
- **Stealthy**: Maintains high test accuracy while achieving backdoor

## Installation

### Requirements

```bash
# Python 3.8+
pip install numpy pandas scikit-learn torch tqdm
```

### Verify Installation

```powershell
python -c "import numpy, pandas, sklearn, torch; print('All packages installed!')"
```

## Complete Workflow

### Step 1: Prepare WISDM Dataset

Process the raw WISDM data into windowed samples:

```powershell
python data\load_wisdm_data.py
```

**What this does:**
- Loads raw accelerometer data from `WISDM_ar_v1.1_raw.txt`
- Creates sliding windows (80 samples, 50% overlap)
- Splits into user-independent train/test sets (80/20)
- Normalizes data using training set statistics
- Computes per-channel perturbation budgets
- Saves processed data to `data/wisdm_processed/`

**Expected output:**
```
Training samples: ~35,000-40,000
Test samples: ~8,000-10,000
Window shape: (80, 3)
```

**Generated files:**
- `X_train.npy`, `X_test.npy` - Normalized windowed samples (N, 80, 3)
- `y_train.npy`, `y_test.npy` - Activity labels (0-5)
- `eps_per_channel.npy` - Perturbation budget for each axis
- `normalization_mean.npy`, `normalization_std.npy` - Statistics

---

### Step 2: Train Surrogate Model

Train a clean LSTM model that will be used to generate poisons:

```powershell
python train_wisdm_surrogate.py
```

**What this does:**
- Creates a 2-layer LSTM with 64 hidden units
- Trains for 30 epochs on clean WISDM data
- Uses learning rate scheduling
- Saves best model based on test accuracy

**Training time:** ~5-10 minutes (CPU), ~2-3 minutes (GPU)

**Expected performance:**
```
Test Accuracy: 85-92%
Classes: Walking, Jogging, Upstairs, Downstairs, Sitting, Standing
```

**Generated file:**
- `models/wisdm_surrogate.pth` - Trained model weights

**Model architecture:**
```
Input: (batch, 80, 3)
  ↓
BatchNorm1d
  ↓
LSTM (2 layers, hidden=64)
  ↓
Classifier (Linear + ReLU + Dropout)
  ↓
Output: (batch, 6)
```

---

### Step 3: Compute Subspace Basis

Compute PCA-based subspace for constrained perturbations:

```powershell
python compute_wisdm_subspace.py
```

**What this does:**
- Computes global PCA on flattened training data
- Extracts 60-dimensional subspace (explains ~80% variance)
- Computes basis matrix U (240×60) and projection M (60×240)
- Optionally computes per-class subspaces for targeted attacks

**Computation time:** ~1-2 minutes

**Generated files:**

`wisdm_subspace/` (global subspace):
- `U.npy` - Basis matrix (240, 60)
- `M.npy` - Projection matrix (60, 240)
- `mu_global.npy` - Global mean (240,)
- `metadata.json` - Variance explained, dimensions

`wisdm_subspace_groups/` (per-class, optional):
- `group_0_U.npy`, `group_0_M.npy`, `group_0_mu.npy` - Walking
- `group_1_U.npy`, `group_1_M.npy`, `group_1_mu.npy` - Jogging
- ... (for all 6 classes)
- `metadata.json` - Per-class statistics

**Why subspace?**
- Constrains perturbations to natural data variations
- Reduces optimization dimensions (240 → 60)
- Makes poisons harder to detect

---

### Step 4: Run Backdoor Attack

Execute the complete attack pipeline:

```powershell
python wisdm_attack_main.py
```

**What this does:**
1. Loads processed data and subspace matrices
2. Selects target sample (default: idx=10)
3. Chooses seed samples from target's class
4. Optimizes poisons using feature collision
5. Constructs poisoned training set
6. Trains clean and poisoned models
7. Evaluates attack success

**Attack configuration (default):**
```python
num_poisons = 200          # Number of poison samples
target_idx = 10            # Index of target sample
optimization_steps = 1000  # Gradient descent iterations
optimization_lr = 0.01     # Learning rate
```

**Runtime:** ~20-30 minutes (CPU), ~10-15 minutes (GPU)

**Expected results:**
```
Target Sample:
  True activity: Walking
  Clean model prediction: Walking
  Poisoned model prediction: Jogging  ← Attack success!
  Attack success: ✓ YES

Test Set Accuracy:
  Clean model: 87.5%
  Poisoned model: 86.8%
  Accuracy drop: 0.7%      ← Stealthy!
  
Attack Configuration:
  Poisoned samples: 200 (0.5% of training)
  Average perturbation: 0.23
  
Overall Effectiveness: 90/100
```

**Success criteria:**
- ✓ Target misclassified to desired class
- ✓ Accuracy drop < 5% (stealthy)
- ✓ Test accuracy > 80% (functional model)

---

## Customizing the Attack

### Change Target Sample

Edit `wisdm_attack_main.py`:

```python
results = run_wisdm_backdoor_attack(
    target_idx=100,  # Change this to any index
    ...
)
```

### Adjust Number of Poisons

```python
results = run_wisdm_backdoor_attack(
    num_poisons=500,  # More poisons = stronger backdoor
    ...
)
```

### Modify Optimization Parameters

```python
results = run_wisdm_backdoor_attack(
    optimization_steps=1500,  # More steps = better optimization
    optimization_lr=0.02,     # Higher LR = faster convergence
    ...
)
```

### Use Different Model Architecture

Edit `models/wisdm_models.py` to try:
- `WISDMAttentionLSTM` - LSTM with attention mechanism
- `WISDMCNNLSTM` - Hybrid CNN-LSTM model

Then retrain surrogate:
```python
# In train_wisdm_surrogate.py
from models.wisdm_models import WISDMAttentionLSTM

model = WISDMAttentionLSTM(
    input_size=3,
    hidden_size=128,
    num_layers=2,
    ...
)
```

---

## Understanding the Code

### Data Format

```python
X_train.shape = (N, 80, 3)  # N samples, 80 timesteps, 3 channels
y_train.shape = (N,)        # Labels: 0-5
```

**Activity labels:**
- 0 = Walking
- 1 = Jogging
- 2 = Upstairs
- 3 = Downstairs
- 4 = Sitting
- 5 = Standing

### Poison Optimization

The core optimization loop (in `wisdm_poison_optimize.py`):

```python
# 1. Compute perturbation in subspace
delta = U @ alpha.T  # Project latent variables to data space

# 2. Create poison candidates
poisons = seeds + delta

# 3. Feature collision loss
feat_poisons = model.get_features(poisons)
feat_target = model.get_features(target)
loss_feat = 1 - cosine_similarity(feat_poisons, feat_target)

# 4. Regularization
loss_l2 = lambda_l2 * ||delta||^2

# 5. Total loss
loss = loss_feat + loss_l2

# 6. Gradient descent
loss.backward()
optimizer.step()

# 7. Project to subspace periodically
delta_clipped = clip_per_channel(delta)
alpha = delta_clipped @ M.T
```

### Feature Hook

Extract penultimate layer features:

```python
def register_feature_hook(model):
    def hook_fn(module, inp, out):
        global _features
        _features = inp[0]  # Capture input to final layer
    
    model.classifier[-1].register_forward_hook(hook_fn)
```

---

## Troubleshooting

### Issue: "FileNotFoundError: WISDM_ar_v1.1_raw.txt"

**Solution:** Ensure the raw WISDM dataset is in the correct location:
```
data/WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt
```

### Issue: Low attack success rate (<50%)

**Solutions:**
1. Increase number of poisons: `num_poisons=400`
2. More optimization steps: `optimization_steps=1500`
3. Add more target copies in poisoned training set
4. Increase learning rate: `optimization_lr=0.02`

### Issue: High accuracy drop (>5%)

**Solutions:**
1. Reduce number of poisons
2. Increase L2 regularization: `lambda_l2=0.05`
3. Train poisoned model for more epochs
4. Use smaller perturbation budget (reduce `k` in data loader)

### Issue: Out of memory

**Solutions:**
1. Reduce batch size: `batch_size=128`
2. Use CPU instead of GPU: Set `DEVICE = "cpu"`
3. Process data in smaller batches
4. Reduce model size: `hidden_size=32`

### Issue: Slow training/optimization

**Solutions:**
1. Use GPU if available (10x speedup)
2. Reduce optimization steps: `optimization_steps=500`
3. Increase batch size: `batch_size=512`
4. Use fewer poisons initially for testing

---

## Advanced Usage

### Batch Attack on Multiple Targets

```python
for target_idx in [10, 50, 100, 200, 500]:
    print(f"\nAttacking target {target_idx}...")
    results = run_wisdm_backdoor_attack(
        target_idx=target_idx,
        num_poisons=200,
        ...
    )
    # Save results for analysis
```

### Cross-Activity Attack Matrix

Test attack success across all activity pairs:

```python
import itertools

for target_class, attack_class in itertools.product(range(6), range(6)):
    if target_class == attack_class:
        continue
    
    # Find sample from target_class
    target_idx = np.where(y_train == target_class)[0][0]
    
    # Run attack to misclassify as attack_class
    ...
```

### Evaluate Defense Mechanisms

Test robustness against:
1. **Outlier removal**: Remove samples with high loss
2. **Activation clustering**: Cluster features, remove outliers
3. **Data sanitization**: Filter based on perturbation norms

---

## Performance Metrics

### Attack Metrics

- **Attack Success Rate (ASR)**: % of target samples misclassified
- **Clean Accuracy (CA)**: Test accuracy on clean data
- **Accuracy Drop (AD)**: CA_clean - CA_poisoned
- **Stealthiness**: AD < 5%

### Perturbation Metrics

- **L2 norm**: Average ||poison - seed||₂
- **L∞ norm**: Max channel-wise perturbation
- **Perceptual distance**: User study on detectability

---

## Citation

If you use this code, please cite the WISDM dataset paper:

```bibtex
@inproceedings{kwapisz2011activity,
  title={Activity recognition using cell phone accelerometers},
  author={Kwapisz, Jennifer R and Weiss, Gary M and Moore, Samuel A},
  booktitle={Proceedings of the Fourth International Workshop on Knowledge Discovery from Sensor Data},
  pages={10--18},
  year={2011}
}
```

---

## Files Summary

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `load_wisdm_data.py` | Data preprocessing | Raw .txt file | Processed .npy files |
| `train_wisdm_surrogate.py` | Train surrogate | Processed data | Model .pth |
| `compute_wisdm_subspace.py` | PCA subspace | Training data | U, M, mu matrices |
| `wisdm_poison_optimize.py` | Poison generation | Seeds, target, model | Optimized poisons |
| `wisdm_attack_main.py` | Complete pipeline | All above | Attack results |

---

## Quick Start (TL;DR)

```powershell
# 1. Process data
python data\load_wisdm_data.py

# 2. Train surrogate
python train_wisdm_surrogate.py

# 3. Compute subspace
python compute_wisdm_subspace.py

# 4. Run attack
python wisdm_attack_main.py
```

Total time: ~30-45 minutes

Expected: 80-95% attack success, <5% accuracy drop

---

## License

This code is for research and educational purposes only. The WISDM dataset has its own license - see the readme.txt in the data folder for details.

## Contact

For questions or issues, please open a GitHub issue or contact the project maintainers.
