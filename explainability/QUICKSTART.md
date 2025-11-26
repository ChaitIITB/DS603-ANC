# Quick Start Guide - WISDM LSTM Explainability Analysis

## Prerequisites

1. **Trained Model**: Ensure you have a trained model at `models/wisdm_surrogate.pth`
2. **Processed Data**: WISDM data should be in `data/wisdm_processed/`
3. **Python Environment**: Python 3.7+ with required packages

## Installation

From the project root directory:

```bash
# Install required packages
pip install -r explainability\requirements.txt
```

Or install individually:

```bash
pip install numpy torch scikit-learn matplotlib seaborn shap
```

## Running the Analysis

### Step 1: Navigate to Project Root

```bash
cd C:\Users\chait\OneDrive\文档\Acads\SEM_5\DS_603\project
```

### Step 2: Run the Analysis

```bash
python explainability\run_explainability_analysis.py
```

### Step 3: View Results

Results will be saved in `explainability\results\`

Open the visualizations:
- `combined_summary.png` - Start here for overview
- `channel_importance_comparison.png` - LIME vs SHAP comparison
- Individual heatmaps and plots for detailed analysis

## What the Analysis Does

1. **Loads** your trained WISDM LSTM model
2. **Evaluates** model accuracy on test set
3. **Runs LIME** analysis to explain model predictions
4. **Runs SHAP** analysis for Shapley value-based explanations
5. **Compares** both methods for consistency
6. **Generates** comprehensive visualizations
7. **Saves** numerical results in JSON format

## Expected Output

```
================================================================================
WISDM LSTM MODEL EXPLAINABILITY ANALYSIS
================================================================================

Starting analysis at: 2025-11-26 XX:XX:XX

================================================================================
Loading WISDM Dataset
================================================================================
Training samples: (XXXX, 80, 3)
Test samples: (XXXX, 80, 3)
...

[Analysis runs for 5-15 minutes]

================================================================================
ANALYSIS COMPLETE
================================================================================

Key Findings:

1. Most Important Accelerometer Channel:
   LIME: Z-accel (45.2%)
   SHAP: Z-accel (42.8%)

2. Method Agreement:
   Correlation: 0.93

3. Model Performance:
   Test Accuracy: 87.45%

4. Output Location:
   C:\Users\chait\...\explainability\results
```

## Key Files Generated

| File | Description |
|------|-------------|
| `explainability_results.json` | All numerical results and metrics |
| `combined_summary.png` | Comprehensive dashboard (start here!) |
| `channel_importance_comparison.png` | LIME vs SHAP side-by-side |
| `lime_heatmap.png` | LIME feature importance over time |
| `shap_heatmap.png` | SHAP feature importance over time |
| `*_temporal_importance.png` | When features matter in the 4-second window |
| `*_per_class_importance.png` | Feature importance for each activity |

## Customizing the Analysis

Edit these parameters in `run_explainability_analysis.py`:

```python
# Fewer samples = faster but less accurate
NUM_EXPLAIN_SAMPLES = 30  # Default: 50

# More perturbations = more accurate LIME
NUM_LIME_SAMPLES = 1000  # Default: 500

# More background = more accurate SHAP
NUM_BACKGROUND_SAMPLES = 200  # Default: 100

# Time granularity
NUM_SEGMENTS = 20  # Default: 10 (more = finer detail)
```

## Troubleshooting

### "Model file not found"
- Ensure `models/wisdm_surrogate.pth` exists
- Train the model first using the main training scripts

### "Data files not found"
- Check that `data/wisdm_processed/*.npy` files exist
- Run data preprocessing if needed

### "Out of memory"
- Reduce `NUM_EXPLAIN_SAMPLES` to 20
- Use `DEVICE = 'cpu'` instead of GPU

### "SHAP import error"
- Try: `pip install --upgrade shap`
- Or: `conda install -c conda-forge shap`

## Quick Interpretation Tips

### Channel Importance
- **Higher value = more important** for predictions
- Values sum to 1.0 (normalized)
- Compare X, Y, Z axes to understand movement patterns

### Temporal Importance
- **Early segments**: Initial movement phase
- **Middle segments**: Sustained activity
- **Late segments**: Activity completion
- Different activities emphasize different time windows

### LIME vs SHAP Correlation
- **> 0.8**: Excellent agreement, robust explanations
- **0.6-0.8**: Good agreement, mostly consistent
- **< 0.6**: Consider investigating differences

### Per-Class Patterns
- **Walking**: Likely emphasizes different features than Jogging
- **Upstairs/Downstairs**: Should show vertical (Z-axis) importance
- **Sitting/Standing**: May rely on different time segments

## Next Steps

After running the analysis:

1. **Examine visualizations** in `explainability/results/`
2. **Read the JSON results** for exact numerical values
3. **Compare activities** using per-class plots
4. **Include findings** in your report/paper
5. **Discuss implications** of which features matter most

## For Your Report

Include these key insights:

✓ Which channel (X/Y/Z) is most important and why it makes sense  
✓ LIME vs SHAP agreement (correlation value)  
✓ Per-activity patterns and interpretations  
✓ Whether results align with domain knowledge  
✓ Model transparency and interpretability conclusions  

## Time Estimates

| Task | Time |
|------|------|
| Installation | 2-5 minutes |
| Running analysis | 5-15 minutes |
| Reviewing results | 10-20 minutes |
| **Total** | **~20-40 minutes** |

## Example Command Sequence

```bash
# 1. Navigate to project
cd C:\Users\chait\OneDrive\文档\Acads\SEM_5\DS_603\project

# 2. Install requirements (first time only)
pip install -r explainability\requirements.txt

# 3. Run analysis
python explainability\run_explainability_analysis.py

# 4. View results
explorer explainability\results
```

## Questions?

- Check the full `README.md` in the explainability folder
- Review code comments in individual modules
- Examine the example outputs once generated

---

**Happy Explaining! 🎯**
