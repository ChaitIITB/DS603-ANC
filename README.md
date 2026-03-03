# Clean-Label Backdoor Attacks on Human Activity Recognition

A research framework for investigating clean-label poisoning attacks on time-series Human Activity Recognition (HAR) systems, combining explainability methods (LIME/SHAP) with targeted adversarial perturbations.

## Overview

Clean-label backdoor attacks represent a stealthy form of data poisoning where adversaries inject malicious training samples **while preserving their true labels**, making detection through label verification impossible. This project demonstrates how explainability tools can be weaponized to identify discriminative features and craft targeted attacks.

### Key Findings

- **Linear models** on WISDM achieve up to **22.7% Attack Success Rate (ASR)** with clean-label attacks
- **Deep learning models** on UCI-HAR prove more robust with ASR below 4%
- **Y-acceleration** contributes **46.2%** of classification importance (SHAP analysis)
- Feature collision attacks exploit learned representations to maximize perturbation efficiency

## Project Structure

```
├── attacks/                 # Attack implementations
│   ├── clean_label_attack.py    # Clean-label & feature collision attacks
│   └── linear_attacks.py        # Attacks for linear models
├── models/                  # Model architectures
│   └── models.py               # Linear, CNN, LSTM, SVM, Ridge, Logistic
├── explainability/          # Feature importance analysis
│   └── feature_importance.py   # LIME & SHAP implementations
├── data/                    # Dataset loaders
│   ├── load_uci_har.py         # UCI HAR dataset
│   └── load_wisdm_data.py      # WISDM dataset
├── Experiments/             # Experiment runners
│   ├── run_experiments.py      # Main experiment pipeline
│   └── results/                # Saved models & metrics
├── defences/                # Defense mechanisms (WIP)
└── notebooks/               # Visualization & analysis
```

## Datasets

| Dataset | Samples | Channels | Classes | Activities |
|---------|---------|----------|---------|------------|
| **UCI-HAR** | 10,299 | 9 | 6 | Walking, Upstairs, Downstairs, Sitting, Standing, Laying |
| **WISDM** | 1,098,207 | 3 | 6 | Walking, Jogging, Upstairs, Downstairs, Sitting, Standing |

## Models

### Deep Learning (PyTorch)
- **LinearModel**: Fully connected layers with batch normalization
- **CNNModel**: 1D Convolutional Neural Network with global average pooling
- **LSTMModel**: Bidirectional LSTM with attention mechanism

### Traditional ML (scikit-learn)
- Logistic Regression
- Linear SVM
- Ridge Classifier
- SGD Classifier

## Attack Methods

### 1. Clean-Label Attack
Uses LIME/SHAP feature importance to craft perturbations targeting the most discriminative features while maintaining correct labels.

### 2. Feature Collision Attack
Optimizes perturbations to make poisoned samples collide with target class samples in feature space.

### 3. Gradient-Based Attack
Exploits model gradients to find optimal perturbation directions (white-box).

### 4. Label-Flip Attack (Baseline)
Traditional backdoor attack that flips labels—included for comparison.

## Installation

```bash
# Clone the repository
git clone https://github.com/ChaitIITB/DS603-ANC.git
cd DS603-ANC

# Create virtual environment
python -m venv ds603
source ds603/bin/activate  # Linux/Mac
# or
.\ds603\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install torch numpy scikit-learn tqdm matplotlib
```

## Usage

### Run Full Experiment Pipeline

```bash
cd Experiments
python run_experiments.py
```

This will:
1. Load UCI-HAR and WISDM datasets
2. Train models (Linear, CNN, LSTM)
3. Compute LIME/SHAP feature importance
4. Execute clean-label backdoor attacks
5. Evaluate Attack Success Rate (ASR)

### Run Linear Model Experiments

```bash
python run_linear_experiments.py
```

### Custom Attack

```python
from attacks.clean_label_attack import CleanLabelAttack
from explainability.feature_importance import combined_importance_analysis

# Compute feature importance
importance = combined_importance_analysis(model, X_train, y_train)

# Initialize attack
attack = CleanLabelAttack(
    model=model,
    importance_matrix=importance,
    eps_per_channel=eps,
    target_class=0,
    trigger_strength=0.8
)

# Create poisoned dataset
X_poisoned, y_poisoned = attack.create_poisoned_dataset(
    X_train, y_train, poison_rate=0.1
)
```

## Results

### UCI-HAR Dataset

| Model | Clean Accuracy | Clean-Label ASR | Feature Collision ASR |
|-------|---------------|-----------------|----------------------|
| CNN | 91.2% | 3.8% | 4.1% |
| LSTM | 89.7% | 2.9% | 3.5% |
| Linear | 87.4% | 6.2% | 7.8% |

### WISDM Dataset

| Model | Clean Accuracy | Clean-Label ASR | Feature Collision ASR |
|-------|---------------|-----------------|----------------------|
| Logistic | 52.1% | 85.4% | 77.9% |
| Linear SVM | 50.9% | 89.9% | 79.4% |
| Ridge | 50.1% | 92.7% | - |

## Tech Stack

- **PyTorch** - Deep learning models
- **scikit-learn** - Traditional ML models
- **NumPy** - Numerical computing
- **LIME/SHAP** - Explainability methods
- **tqdm** - Progress bars
- **Matplotlib** - Visualization

## References

- [BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain](https://arxiv.org/abs/1708.06733)
- [Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks](https://arxiv.org/abs/1804.00792)
- [BackTime: Backdoor Attacks on Multivariate Time Series Classification](https://arxiv.org/abs/2307.12335)
- [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- [WISDM Dataset](https://www.cis.fordham.edu/wisdm/dataset.php)

## Authors

- Nikhil
- Chaitanya  
- Anasmit

## Course

**DS603: Adversarial & Novel Challenges in Machine Learning**  
IIT Bombay, Fall 2025

## License

MIT License
