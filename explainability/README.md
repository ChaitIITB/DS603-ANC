# WISDM LSTM Model Explainability Analysis

This module provides comprehensive explainability analysis for the WISDM Activity Recognition LSTM model using **LIME** (Local Interpretable Model-agnostic Explanations) and **SHAP** (SHapley Additive exPlanations).

## Overview

The explainability analysis helps understand:
- **Which accelerometer channels** (X, Y, Z) are most important for activity classification
- **Which time segments** within the 4-second window are most critical
- **How different methods** (LIME vs SHAP) compare in their interpretations
- **Per-activity patterns** showing what features matter for each activity class

## Directory Structure

```
explainability/
├── __init__.py                          # Module initialization
├── lime_explainer.py                    # LIME implementation for time series
├── shap_explainer.py                    # SHAP implementation for LSTM
├── visualizations.py                    # Plotting utilities
├── run_explainability_analysis.py       # Main analysis pipeline
├── README.md                            # This file
└── results/                             # Generated outputs (created on run)
    ├── explainability_results.json      # Numerical results
    ├── lime_importance_matrix.npy       # Raw LIME data
    ├── shap_importance_matrix.npy       # Raw SHAP data
    ├── lime_heatmap.png                 # LIME feature importance heatmap
    ├── shap_heatmap.png                 # SHAP feature importance heatmap
    ├── channel_importance_comparison.png # LIME vs SHAP comparison
    ├── lime_temporal_importance.png     # Temporal patterns (LIME)
    ├── shap_temporal_importance.png     # Temporal patterns (SHAP)
    ├── lime_per_class_importance.png    # Per-activity analysis (LIME)
    ├── shap_per_class_importance.png    # Per-activity analysis (SHAP)
    └── combined_summary.png             # Comprehensive summary figure
```

## Requirements

Ensure the following packages are installed:

```bash
pip install numpy torch scikit-learn matplotlib seaborn shap
```

## How to Run

### 1. Basic Analysis (Recommended)

Simply run the main analysis script from the project root directory:

```bash
cd C:\Users\chait\OneDrive\文档\Acads\SEM_5\DS_603\project
python explainability\run_explainability_analysis.py
```

This will:
1. Load the trained WISDM LSTM model
2. Run LIME analysis on test samples
3. Run SHAP analysis on test samples
4. Generate all visualizations
5. Save results to `explainability/results/`

**Expected runtime:** 5-15 minutes depending on your hardware

### 2. Custom Configuration

Edit the configuration section in `run_explainability_analysis.py`:

```python
# Configuration
DATA_DIR = 'data/wisdm_processed'
MODEL_PATH = 'models/wisdm_surrogate.pth'
OUTPUT_DIR = 'explainability/results'

# Analysis parameters
NUM_BACKGROUND_SAMPLES = 100  # For SHAP
NUM_LIME_SAMPLES = 500        # Number of perturbations for LIME
NUM_EXPLAIN_SAMPLES = 50      # Number of test samples to explain
NUM_SEGMENTS = 10             # Time segments for LIME
```

### 3. Using Individual Explainers

You can also use the explainers independently in your own scripts:

#### LIME Example:

```python
from explainability.lime_explainer import TimeSeriesLIME
from models.wisdm_models import WISDMActivityLSTM
import numpy as np
import torch

# Load model and data
model = WISDMActivityLSTM(input_size=3, hidden_size=64, num_layers=2)
model.load_state_dict(torch.load('models/wisdm_surrogate.pth'))
model.eval()

X_test = np.load('data/wisdm_processed/X_test.npy')

# Create LIME explainer
lime = TimeSeriesLIME(model, device='cuda', num_samples=500, num_segments=10)

# Explain a single sample
explanation = lime.explain_instance(X_test[0])
print(f"Importance matrix shape: {explanation['importance_matrix'].shape}")
print(f"Target class: {explanation['target_class']}")
print(f"R² score: {explanation['r2_score']:.4f}")
```

#### SHAP Example:

```python
from explainability.shap_explainer import TimeSeriesSHAP
from models.wisdm_models import WISDMActivityLSTM
import numpy as np
import torch

# Load model and data
model = WISDMActivityLSTM(input_size=3, hidden_size=64, num_layers=2)
model.load_state_dict(torch.load('models/wisdm_surrogate.pth'))
model.eval()

X_train = np.load('data/wisdm_processed/X_train.npy')
X_test = np.load('data/wisdm_processed/X_test.npy')

# Select background samples
background_data = X_train[np.random.choice(len(X_train), 100, replace=False)]

# Create SHAP explainer
shap_explainer = TimeSeriesSHAP(model, background_data, device='cuda', method='gradient')

# Explain a single sample
explanation = shap_explainer.explain_instance(X_test[0])
print(f"SHAP values shape: {explanation['shap_values'].shape}")
print(f"Target class: {explanation['target_class']}")
```

## Understanding the Results

### 1. Channel Importance

Shows which accelerometer axis (X, Y, Z) is most important:
- **X-accel**: Side-to-side movement
- **Y-accel**: Forward-backward movement  
- **Z-accel**: Up-down movement (vertical)

Higher values indicate that channel contributes more to the model's predictions.

### 2. Temporal Importance

Shows which parts of the 4-second window are most critical:
- **Early segments**: Initial movement patterns
- **Middle segments**: Sustained activity patterns
- **Late segments**: Ending movement patterns

Different activities may have importance in different time segments.

### 3. LIME vs SHAP Comparison

- **LIME**: Model-agnostic, uses local linear approximation
- **SHAP**: Model-specific, uses game-theoretic Shapley values

High correlation between methods indicates robust, consistent explanations.

### 4. Per-Class Insights

Shows how feature importance varies by activity:
- Walking vs Jogging might emphasize different channels
- Upstairs vs Downstairs might show temporal pattern differences
- Sitting vs Standing might rely more on specific axes

## Output Files

### Visualizations

All PNG files are saved at 300 DPI for publication quality:

- **Heatmaps**: Show importance across time and channels
- **Bar charts**: Compare channel importance
- **Line plots**: Show temporal patterns
- **Summary dashboard**: Combined overview

### Data Files

- `explainability_results.json`: Human-readable summary with all metrics
- `lime_importance_matrix.npy`: Raw LIME importance values (num_segments × channels)
- `shap_importance_matrix.npy`: Raw SHAP values (time_steps × channels)

## Troubleshooting

### Memory Issues

If you run out of memory:

1. Reduce the number of samples:
   ```python
   NUM_EXPLAIN_SAMPLES = 20  # Instead of 50
   NUM_BACKGROUND_SAMPLES = 50  # Instead of 100
   ```

2. Use CPU instead of GPU:
   ```python
   DEVICE = 'cpu'
   ```

### SHAP Installation Issues

If SHAP fails to install:

```bash
pip install --upgrade shap
# Or for Windows users with issues:
conda install -c conda-forge shap
```

### Module Import Errors

Make sure you run from the project root directory:

```bash
cd C:\Users\chait\OneDrive\文档\Acads\SEM_5\DS_603\project
python explainability\run_explainability_analysis.py
```

## Interpreting Results for Your Report

### Key Questions to Answer:

1. **Which features matter most?**
   - Look at channel importance values
   - Check if LIME and SHAP agree

2. **Are the explanations consistent?**
   - Check correlation between LIME and SHAP (>0.8 is good)
   - Look for agreement on most important channel

3. **Activity-specific patterns?**
   - Examine per-class importance plots
   - Identify which channels matter for which activities

4. **Model behavior insights?**
   - Does the model focus on expected features?
   - Are there surprising patterns?

### Sample Interpretation:

> "The explainability analysis reveals that the **Z-axis acceleration** (vertical movement) 
> is the most important feature for activity classification, accounting for 45% of the model's 
> decision-making (LIME) and 42% (SHAP). This makes intuitive sense as activities like 
> 'Upstairs', 'Downstairs', and 'Jogging' have distinct vertical motion patterns. The high 
> correlation (r=0.93) between LIME and SHAP indicates robust, reliable explanations."

## Advanced Usage

### Batch Processing Multiple Models

```python
models = ['wisdm_surrogate.pth', 'wisdm_alternative.pth']

for model_path in models:
    # Load model
    # Run analysis
    # Save results with model name prefix
```

### Focus on Specific Activities

```python
# Explain only walking samples
walking_indices = np.where(y_test == 0)[0]
X_walking = X_test[walking_indices]

lime_results = lime_explainer.explain_batch(X_walking, num_samples=30)
```

### Export for Statistical Analysis

```python
import pandas as pd

# Create DataFrame for statistical testing
df = pd.DataFrame({
    'Channel': CHANNEL_NAMES,
    'LIME_Importance': lime_channel,
    'SHAP_Importance': shap_channel
})

df.to_csv('explainability/results/importance_comparison.csv', index=False)
```

## References

- **LIME Paper**: "Why Should I Trust You?" - Ribeiro et al., 2016
- **SHAP Paper**: "A Unified Approach to Interpreting Model Predictions" - Lundberg & Lee, 2017
- **WISDM Dataset**: Kwapisz et al., "Activity Recognition using Cell Phone Accelerometers", 2010

## Contact & Support

For issues or questions about the explainability analysis:
1. Check the troubleshooting section above
2. Review the example outputs in `explainability/results/`
3. Examine the code comments in individual modules

---

**Note**: This analysis assumes you have already trained the WISDM LSTM model and have the 
model file at `models/wisdm_surrogate.pth`. If not, train the model first using the main 
training scripts in the project.
