"""
Comprehensive Explainability Analysis for WISDM LSTM Model

This script runs both LIME and SHAP analysis on the trained LSTM model
and generates visualizations and reports.

Usage:
    python run_explainability_analysis.py
"""

import os
import sys
import numpy as np
import torch
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.wisdm_models import WISDMActivityLSTM
from explainability.lime_explainer import (
    TimeSeriesLIME, 
    analyze_channel_importance, 
    analyze_temporal_importance
)
from explainability.shap_explainer import (
    TimeSeriesSHAP,
    analyze_channel_importance_shap,
    analyze_temporal_importance_shap,
    compute_feature_importance
)
from explainability.visualizations import create_summary_dashboard


# Configuration
DATA_DIR = 'data/wisdm_processed'
MODEL_PATH = 'models/wisdm_surrogate.pth'
OUTPUT_DIR = 'explainability/results'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Analysis parameters
NUM_BACKGROUND_SAMPLES = 100  # For SHAP
NUM_LIME_SAMPLES = 500        # Number of perturbations for LIME
NUM_EXPLAIN_SAMPLES = 50      # Number of test samples to explain
NUM_SEGMENTS = 10             # Time segments for LIME

# Activity names
ACTIVITY_NAMES = ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
CHANNEL_NAMES = ['X-accel', 'Y-accel', 'Z-accel']


def load_data():
    """Load WISDM dataset."""
    print("\n" + "=" * 80)
    print("Loading WISDM Dataset")
    print("=" * 80)
    
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    
    print(f"Training samples: {X_train.shape}")
    print(f"Test samples: {X_test.shape}")
    print(f"Input shape: (time_steps={X_train.shape[1]}, channels={X_train.shape[2]})")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    return X_train, y_train, X_test, y_test


def load_model():
    """Load trained WISDM LSTM model."""
    print("\n" + "=" * 80)
    print("Loading LSTM Model")
    print("=" * 80)
    
    model = WISDMActivityLSTM(
        input_size=3,
        hidden_size=64,
        num_layers=2,
        num_classes=6,
        dropout_rate=0.3
    )
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded from: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Total parameters: {num_params:,}")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model accuracy."""
    print("\n" + "=" * 80)
    print("Evaluating Model Performance")
    print("=" * 80)
    
    correct = 0
    total = len(X_test)
    
    with torch.no_grad():
        for i in range(total):
            x = torch.from_numpy(X_test[i]).float().unsqueeze(0).to(DEVICE)
            logits = model(x)
            pred = logits.argmax(1).item()
            if pred == y_test[i]:
                correct += 1
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    
    return accuracy


def run_lime_analysis(model, X_train, X_test, y_test):
    """Run LIME explainability analysis."""
    print("\n" + "=" * 80)
    print("Running LIME Analysis")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Number of perturbations: {NUM_LIME_SAMPLES}")
    print(f"  Number of time segments: {NUM_SEGMENTS}")
    print(f"  Samples to explain: {NUM_EXPLAIN_SAMPLES}")
    
    # Create LIME explainer
    lime_explainer = TimeSeriesLIME(
        model=model,
        device=DEVICE,
        num_samples=NUM_LIME_SAMPLES,
        num_segments=NUM_SEGMENTS
    )
    
    # Explain batch of samples
    print("\nExplaining test samples with LIME...")
    lime_results = lime_explainer.explain_batch(
        X_test, 
        y_test, 
        num_samples=NUM_EXPLAIN_SAMPLES
    )
    
    # Analyze results
    print("\n" + "-" * 80)
    print("LIME Results Summary")
    print("-" * 80)
    
    channel_importance = analyze_channel_importance(lime_results)
    print(f"\nChannel Importance:")
    for i, name in enumerate(CHANNEL_NAMES):
        print(f"  {name:12s}: {channel_importance[i]:.4f} ({channel_importance[i]*100:.1f}%)")
    
    temporal_importance = analyze_temporal_importance(lime_results)
    print(f"\nTop 3 Most Important Time Segments:")
    top_segments = np.argsort(temporal_importance)[-3:][::-1]
    for rank, seg in enumerate(top_segments, 1):
        print(f"  {rank}. Segment {seg}: {temporal_importance[seg]:.4f}")
    
    return lime_results


def run_shap_analysis(model, X_train, X_test, y_test):
    """Run SHAP explainability analysis."""
    print("\n" + "=" * 80)
    print("Running SHAP Analysis")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Background samples: {NUM_BACKGROUND_SAMPLES}")
    print(f"  Samples to explain: {NUM_EXPLAIN_SAMPLES}")
    print(f"  Method: GradientExplainer")
    
    # Select background samples
    background_indices = np.random.choice(len(X_train), NUM_BACKGROUND_SAMPLES, replace=False)
    background_data = X_train[background_indices]
    
    print(f"\nCreating SHAP explainer...")
    shap_explainer = TimeSeriesSHAP(
        model=model,
        background_data=background_data,
        device=DEVICE,
        method='gradient'
    )
    
    # Explain batch of samples
    print("\nExplaining test samples with SHAP...")
    shap_results = shap_explainer.explain_batch(
        X_test,
        y_test,
        num_samples=NUM_EXPLAIN_SAMPLES
    )
    
    # Analyze results
    print("\n" + "-" * 80)
    print("SHAP Results Summary")
    print("-" * 80)
    
    channel_importance = analyze_channel_importance_shap(shap_results)
    print(f"\nChannel Importance:")
    for i, name in enumerate(CHANNEL_NAMES):
        print(f"  {name:12s}: {channel_importance[i]:.4f} ({channel_importance[i]*100:.1f}%)")
    
    temporal_importance = analyze_temporal_importance_shap(shap_results, num_segments=NUM_SEGMENTS)
    print(f"\nTop 3 Most Important Time Segments:")
    top_segments = np.argsort(temporal_importance)[-3:][::-1]
    for rank, seg in enumerate(top_segments, 1):
        print(f"  {rank}. Segment {seg}: {temporal_importance[seg]:.4f}")
    
    # Feature importance
    feature_imp = compute_feature_importance(shap_results)
    print(f"\nTop 5 Most Important Features (time-channel pairs):")
    for i, (t, c, val) in enumerate(feature_imp['top_features'][:5], 1):
        print(f"  {i}. Time step {t:2d}, {CHANNEL_NAMES[c]:10s}: {val:.6f}")
    
    return shap_results


def compare_methods(lime_results, shap_results):
    """Compare LIME and SHAP results."""
    print("\n" + "=" * 80)
    print("Comparing LIME and SHAP Results")
    print("=" * 80)
    
    lime_channel = analyze_channel_importance(lime_results)
    shap_channel = analyze_channel_importance_shap(shap_results)
    
    print(f"\nChannel Importance Comparison:")
    print(f"{'Channel':<12} {'LIME':>10} {'SHAP':>10} {'Difference':>12}")
    print("-" * 48)
    for i, name in enumerate(CHANNEL_NAMES):
        diff = abs(lime_channel[i] - shap_channel[i])
        print(f"{name:<12} {lime_channel[i]:>10.4f} {shap_channel[i]:>10.4f} {diff:>12.4f}")
    
    # Correlation between methods
    correlation = np.corrcoef(lime_channel, shap_channel)[0, 1]
    print(f"\nCorrelation between LIME and SHAP: {correlation:.4f}")
    
    # Agreement
    lime_top = np.argmax(lime_channel)
    shap_top = np.argmax(shap_channel)
    
    print(f"\nMost Important Channel:")
    print(f"  LIME: {CHANNEL_NAMES[lime_top]}")
    print(f"  SHAP: {CHANNEL_NAMES[shap_top]}")
    print(f"  Agreement: {'✓ YES' if lime_top == shap_top else '✗ NO'}")


def save_results(lime_results, shap_results, model_accuracy):
    """Save analysis results to JSON."""
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    lime_channel = analyze_channel_importance(lime_results)
    shap_channel = analyze_channel_importance_shap(shap_results)
    lime_temporal = analyze_temporal_importance(lime_results)
    shap_temporal = analyze_temporal_importance_shap(shap_results, num_segments=NUM_SEGMENTS)
    
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_path': MODEL_PATH,
            'model_accuracy': float(model_accuracy),
            'device': DEVICE,
            'num_explain_samples': NUM_EXPLAIN_SAMPLES,
        },
        'lime': {
            'channel_importance': {
                name: float(lime_channel[i]) for i, name in enumerate(CHANNEL_NAMES)
            },
            'temporal_importance': lime_temporal.tolist(),
            'num_segments': NUM_SEGMENTS,
            'num_samples': NUM_LIME_SAMPLES,
        },
        'shap': {
            'channel_importance': {
                name: float(shap_channel[i]) for i, name in enumerate(CHANNEL_NAMES)
            },
            'temporal_importance': shap_temporal.tolist(),
            'num_segments': NUM_SEGMENTS,
            'num_background_samples': NUM_BACKGROUND_SAMPLES,
        },
        'comparison': {
            'correlation': float(np.corrcoef(lime_channel, shap_channel)[0, 1]),
            'most_important_channel_lime': CHANNEL_NAMES[np.argmax(lime_channel)],
            'most_important_channel_shap': CHANNEL_NAMES[np.argmax(shap_channel)],
        }
    }
    
    results_path = os.path.join(OUTPUT_DIR, 'explainability_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    # Save raw results as numpy arrays
    np.save(os.path.join(OUTPUT_DIR, 'lime_importance_matrix.npy'), 
            lime_results['avg_importance'])
    np.save(os.path.join(OUTPUT_DIR, 'shap_importance_matrix.npy'), 
            shap_results['avg_shap'])
    
    print(f"Raw data saved to: {OUTPUT_DIR}")


def main():
    """Main analysis pipeline."""
    print("\n" + "=" * 80)
    print("WISDM LSTM MODEL EXPLAINABILITY ANALYSIS")
    print("=" * 80)
    print(f"\nStarting analysis at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load data and model
    X_train, y_train, X_test, y_test = load_data()
    model = load_model()
    
    # Evaluate model
    model_accuracy = evaluate_model(model, X_test, y_test)
    
    # Run LIME analysis
    lime_results = run_lime_analysis(model, X_train, X_test, y_test)
    
    # Run SHAP analysis
    shap_results = run_shap_analysis(model, X_train, X_test, y_test)
    
    # Compare methods
    compare_methods(lime_results, shap_results)
    
    # Save results
    save_results(lime_results, shap_results, model_accuracy)
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("Creating Visualizations")
    print("=" * 80)
    create_summary_dashboard(lime_results, shap_results, OUTPUT_DIR)
    
    # Final summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nKey Findings:")
    
    lime_channel = analyze_channel_importance(lime_results)
    shap_channel = analyze_channel_importance_shap(shap_results)
    
    print(f"\n1. Most Important Accelerometer Channel:")
    print(f"   LIME: {CHANNEL_NAMES[np.argmax(lime_channel)]} ({lime_channel[np.argmax(lime_channel)]*100:.1f}%)")
    print(f"   SHAP: {CHANNEL_NAMES[np.argmax(shap_channel)]} ({shap_channel[np.argmax(shap_channel)]*100:.1f}%)")
    
    print(f"\n2. Method Agreement:")
    correlation = np.corrcoef(lime_channel, shap_channel)[0, 1]
    print(f"   Correlation: {correlation:.4f}")
    
    print(f"\n3. Model Performance:")
    print(f"   Test Accuracy: {model_accuracy*100:.2f}%")
    
    print(f"\n4. Output Location:")
    print(f"   {os.path.abspath(OUTPUT_DIR)}")
    
    print("\n" + "=" * 80)
    print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
