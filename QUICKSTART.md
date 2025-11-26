# Quick Start Guide - WISDM Backdoor Attack

## Prerequisites

- Python 3.8+
- Packages: numpy, pandas, scikit-learn, torch, tqdm

Install packages:
```powershell
pip install numpy pandas scikit-learn torch tqdm
```

## Automated Pipeline (Recommended)

Run the complete pipeline with one command:

```powershell
.\run_wisdm_attack.ps1
```

This script will:
1. Check prerequisites
2. Process WISDM data
3. Train surrogate model
4. Compute subspace
5. Run backdoor attack

**Total time:** 30-45 minutes

## Manual Execution

If you prefer to run steps individually:

### Step 1: Process Data
```powershell
python data\load_wisdm_data.py
```

### Step 2: Train Surrogate
```powershell
python train_wisdm_surrogate.py
```

### Step 3: Compute Subspace
```powershell
python compute_wisdm_subspace.py
```

### Step 4: Run Attack
```powershell
python wisdm_attack_main.py
```

## Expected Results

After running the attack, you should see:

```
Target Sample:
  True activity: Walking
  Poisoned model prediction: Jogging  ← Attack works!
  Attack success: ✓ YES

Test Set Accuracy:
  Clean model: 87.5%
  Poisoned model: 86.8%    ← Still accurate (stealthy)
  Accuracy drop: 0.7%

Overall Effectiveness: 90/100
```

## Troubleshooting

**Issue:** "WISDM raw data not found"
- Ensure `data/WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt` exists

**Issue:** Out of memory
- Reduce batch size in training scripts
- Use CPU instead of GPU

**Issue:** Low attack success
- Increase `num_poisons` in `wisdm_attack_main.py`
- Increase `optimization_steps`

## Next Steps

1. Try different target samples by changing `target_idx`
2. Experiment with attack parameters (num_poisons, lr, steps)
3. Test different model architectures
4. Implement defense mechanisms

## Full Documentation

See `WISDM_ATTACK_README.md` for complete documentation.

## File Structure

```
data/wisdm_processed/     - Processed data
models/wisdm_surrogate.pth - Surrogate model
wisdm_subspace/           - Subspace matrices
wisdm_attack_main.py      - Main attack script
```

## Support

For issues or questions:
- Check WISDM_ATTACK_README.md
- Review error messages
- Verify all prerequisites are installed
