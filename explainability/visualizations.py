"""
Visualization utilities for LIME and SHAP explainability results.

This module provides comprehensive plotting functions for:
- Feature importance heatmaps
- Channel importance bar charts
- Temporal importance plots
- Per-class comparison plots
- Summary dashboards
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


# Activity names for WISDM
ACTIVITY_NAMES = ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
CHANNEL_NAMES = ['X-accel', 'Y-accel', 'Z-accel']
COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']


def plot_importance_heatmap(importance_matrix, title="Feature Importance", 
                            channel_names=None, figsize=(12, 6), 
                            save_path=None):
    """
    Plot heatmap of importance values across time and channels.
    
    Args:
        importance_matrix: (num_segments, num_channels) or (time_steps, num_channels)
        title: Plot title
        channel_names: List of channel names
        figsize: Figure size
        save_path: Path to save figure
    """
    if channel_names is None:
        channel_names = CHANNEL_NAMES
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create custom colormap (white to red for positive values)
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    sns.heatmap(
        importance_matrix.T,
        ax=ax,
        cmap=cmap,
        center=0,
        yticklabels=channel_names,
        cbar_kws={'label': 'Importance'},
        linewidths=0.5,
        linecolor='lightgray'
    )
    
    ax.set_xlabel('Time Segment' if importance_matrix.shape[0] < 20 else 'Time Step')
    ax.set_ylabel('Channel')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_channel_importance(channel_importance, channel_names=None, 
                            title="Channel Importance", figsize=(8, 5),
                            save_path=None):
    """
    Plot bar chart of channel importance.
    
    Args:
        channel_importance: (num_channels,) array
        channel_names: List of channel names
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    if channel_names is None:
        channel_names = CHANNEL_NAMES
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors_bar = ['#e74c3c', '#3498db', '#2ecc71']
    bars = ax.bar(channel_names, channel_importance, color=colors_bar, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(channel_importance) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_temporal_importance(temporal_importance, title="Temporal Importance",
                             figsize=(12, 4), save_path=None):
    """
    Plot line chart of temporal importance across segments.
    
    Args:
        temporal_importance: (num_segments,) array
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    segments = np.arange(len(temporal_importance))
    
    ax.plot(segments, temporal_importance, marker='o', linewidth=2, 
            markersize=8, color='#3498db', label='Importance')
    ax.fill_between(segments, 0, temporal_importance, alpha=0.3, color='#3498db')
    
    ax.set_xlabel('Time Segment', fontsize=12)
    ax.set_ylabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xticks(segments)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_per_class_importance(per_class_importance, channel_names=None,
                              title="Per-Class Channel Importance",
                              figsize=(12, 6), save_path=None):
    """
    Plot grouped bar chart comparing channel importance across classes.
    
    Args:
        per_class_importance: Dict {class_id: importance_array}
        channel_names: List of channel names
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    if channel_names is None:
        channel_names = CHANNEL_NAMES
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Aggregate per-class importance by channel
    classes = sorted(per_class_importance.keys())
    num_classes = len(classes)
    num_channels = len(channel_names)
    
    # Compute average importance per channel for each class
    class_channel_importance = np.zeros((num_classes, num_channels))
    
    for i, cls in enumerate(classes):
        importance_matrix = per_class_importance[cls]  # (segments, channels)
        class_channel_importance[i] = np.sum(np.abs(importance_matrix), axis=0)
    
    # Normalize each class
    for i in range(num_classes):
        total = np.sum(class_channel_importance[i])
        if total > 0:
            class_channel_importance[i] /= total
    
    # Plot grouped bars
    x = np.arange(num_classes)
    width = 0.25
    
    for i, channel in enumerate(channel_names):
        offset = (i - 1) * width
        ax.bar(x + offset, class_channel_importance[:, i], width, 
               label=channel, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Activity Class', fontsize=12)
    ax.set_ylabel('Normalized Importance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([ACTIVITY_NAMES[c] for c in classes], rotation=45, ha='right')
    ax.legend(title='Channel', loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_comparison_lime_shap(lime_results, shap_results, 
                              figsize=(14, 5), save_path=None):
    """
    Compare LIME and SHAP channel importance side by side.
    
    Args:
        lime_results: Results from LIME explain_batch
        shap_results: Results from SHAP explain_batch
        figsize: Figure size
        save_path: Path to save figure
    """
    # Compute channel importance
    from explainability.lime_explainer import analyze_channel_importance
    from explainability.shap_explainer import analyze_channel_importance_shap
    
    lime_channel = analyze_channel_importance(lime_results)
    shap_channel = analyze_channel_importance_shap(shap_results)
    
    channel_names = CHANNEL_NAMES
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # LIME
    colors_bar = ['#e74c3c', '#3498db', '#2ecc71']
    axes[0].bar(channel_names, lime_channel, color=colors_bar, alpha=0.7, edgecolor='black')
    axes[0].set_title('LIME Channel Importance', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Importance Score', fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim(0, max(max(lime_channel), max(shap_channel)) * 1.15)
    
    # Add value labels
    for i, v in enumerate(lime_channel):
        axes[0].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # SHAP
    axes[1].bar(channel_names, shap_channel, color=colors_bar, alpha=0.7, edgecolor='black')
    axes[1].set_title('SHAP Channel Importance', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Importance Score', fontsize=11)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim(0, max(max(lime_channel), max(shap_channel)) * 1.15)
    
    # Add value labels
    for i, v in enumerate(shap_channel):
        axes[1].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def create_summary_dashboard(lime_results, shap_results, output_dir='explainability/results'):
    """
    Create comprehensive summary dashboard with multiple plots.
    
    Args:
        lime_results: Results from LIME explain_batch
        shap_results: Results from SHAP explain_batch
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Creating Explainability Summary Dashboard")
    print("=" * 80)
    
    from explainability.lime_explainer import (
        analyze_channel_importance,
        analyze_temporal_importance
    )
    from explainability.shap_explainer import (
        analyze_channel_importance_shap,
        analyze_temporal_importance_shap,
        compute_feature_importance
    )
    
    # 1. LIME Heatmap
    print("\n[1/8] Creating LIME importance heatmap...")
    plot_importance_heatmap(
        lime_results['avg_importance'],
        title="LIME: Feature Importance Across Time Segments",
        save_path=os.path.join(output_dir, 'lime_heatmap.png')
    )
    plt.close()
    
    # 2. SHAP Heatmap
    print("[2/8] Creating SHAP importance heatmap...")
    # Downsample SHAP for visualization (average over windows)
    shap_avg = shap_results['avg_shap']
    window_size = 8
    downsampled_shap = np.array([
        shap_avg[i:i+window_size].mean(axis=0) 
        for i in range(0, len(shap_avg), window_size)
    ])
    
    plot_importance_heatmap(
        downsampled_shap,
        title="SHAP: Feature Importance Across Time Steps",
        save_path=os.path.join(output_dir, 'shap_heatmap.png')
    )
    plt.close()
    
    # 3. Channel importance comparison
    print("[3/8] Creating channel importance comparison...")
    plot_comparison_lime_shap(
        lime_results,
        shap_results,
        save_path=os.path.join(output_dir, 'channel_importance_comparison.png')
    )
    plt.close()
    
    # 4. LIME temporal importance
    print("[4/8] Creating LIME temporal importance plot...")
    lime_temporal = analyze_temporal_importance(lime_results)
    plot_temporal_importance(
        lime_temporal,
        title="LIME: Temporal Importance Across Segments",
        save_path=os.path.join(output_dir, 'lime_temporal_importance.png')
    )
    plt.close()
    
    # 5. SHAP temporal importance
    print("[5/8] Creating SHAP temporal importance plot...")
    shap_temporal = analyze_temporal_importance_shap(shap_results, num_segments=10)
    plot_temporal_importance(
        shap_temporal,
        title="SHAP: Temporal Importance Across Segments",
        save_path=os.path.join(output_dir, 'shap_temporal_importance.png')
    )
    plt.close()
    
    # 6. LIME per-class importance
    print("[6/8] Creating LIME per-class importance plot...")
    if lime_results['per_class_importance']:
        plot_per_class_importance(
            lime_results['per_class_importance'],
            title="LIME: Per-Class Channel Importance",
            save_path=os.path.join(output_dir, 'lime_per_class_importance.png')
        )
        plt.close()
    
    # 7. SHAP per-class importance
    print("[7/8] Creating SHAP per-class importance plot...")
    if shap_results['per_class_shap']:
        plot_per_class_importance(
            shap_results['per_class_shap'],
            title="SHAP: Per-Class Channel Importance",
            save_path=os.path.join(output_dir, 'shap_per_class_importance.png')
        )
        plt.close()
    
    # 8. Combined summary figure
    print("[8/8] Creating combined summary figure...")
    create_combined_summary(lime_results, shap_results, output_dir)
    
    print("\n" + "=" * 80)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 80)


def create_combined_summary(lime_results, shap_results, output_dir):
    """
    Create a single comprehensive summary figure with key insights.
    """
    from explainability.lime_explainer import analyze_channel_importance
    from explainability.shap_explainer import analyze_channel_importance_shap
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('WISDM LSTM Model Explainability Analysis Summary', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. LIME Heatmap
    ax1 = fig.add_subplot(gs[0, :2])
    sns.heatmap(lime_results['avg_importance'].T, ax=ax1, cmap='RdYlBu_r',
                yticklabels=CHANNEL_NAMES, cbar_kws={'label': 'LIME Importance'})
    ax1.set_title('LIME: Feature Importance Heatmap', fontweight='bold')
    ax1.set_xlabel('Time Segment')
    ax1.set_ylabel('Channel')
    
    # 2. LIME Channel Importance
    ax2 = fig.add_subplot(gs[0, 2])
    lime_channel = analyze_channel_importance(lime_results)
    colors_bar = ['#e74c3c', '#3498db', '#2ecc71']
    ax2.bar(CHANNEL_NAMES, lime_channel, color=colors_bar, alpha=0.7, edgecolor='black')
    ax2.set_title('LIME: Channel\nImportance', fontweight='bold')
    ax2.set_ylabel('Importance')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. SHAP Heatmap (downsampled)
    ax3 = fig.add_subplot(gs[1, :2])
    shap_avg = shap_results['avg_shap']
    window_size = 8
    downsampled_shap = np.array([
        shap_avg[i:i+window_size].mean(axis=0) 
        for i in range(0, len(shap_avg), window_size)
    ])
    sns.heatmap(downsampled_shap.T, ax=ax3, cmap='RdYlBu_r',
                yticklabels=CHANNEL_NAMES, cbar_kws={'label': 'SHAP Value'})
    ax3.set_title('SHAP: Feature Importance Heatmap', fontweight='bold')
    ax3.set_xlabel('Time Window')
    ax3.set_ylabel('Channel')
    
    # 4. SHAP Channel Importance
    ax4 = fig.add_subplot(gs[1, 2])
    shap_channel = analyze_channel_importance_shap(shap_results)
    ax4.bar(CHANNEL_NAMES, shap_channel, color=colors_bar, alpha=0.7, edgecolor='black')
    ax4.set_title('SHAP: Channel\nImportance', fontweight='bold')
    ax4.set_ylabel('Importance')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Comparison bar chart
    ax5 = fig.add_subplot(gs[2, :])
    x = np.arange(len(CHANNEL_NAMES))
    width = 0.35
    ax5.bar(x - width/2, lime_channel, width, label='LIME', alpha=0.8, edgecolor='black')
    ax5.bar(x + width/2, shap_channel, width, label='SHAP', alpha=0.8, edgecolor='black')
    ax5.set_xlabel('Channel', fontweight='bold')
    ax5.set_ylabel('Normalized Importance', fontweight='bold')
    ax5.set_title('Method Comparison: Channel Importance', fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(CHANNEL_NAMES)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (lv, sv) in enumerate(zip(lime_channel, shap_channel)):
        ax5.text(i - width/2, lv, f'{lv:.3f}', ha='center', va='bottom', fontsize=8)
        ax5.text(i + width/2, sv, f'{sv:.3f}', ha='center', va='bottom', fontsize=8)
    
    save_path = os.path.join(output_dir, 'combined_summary.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Visualization utilities module")
    print("Import and use the plotting functions in your analysis scripts")
