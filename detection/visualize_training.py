"""
Training Results Visualization Script
Generates comprehensive learning curves and metrics plots
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_training_results(results_dir):
    """
    Load training results from YOLO output directory

    Args:
        results_dir: Path to training results directory

    Returns:
        DataFrame with training metrics
    """
    results_csv = os.path.join(results_dir, 'results.csv')

    if not os.path.exists(results_csv):
        print(f"ERROR: results.csv not found at {results_csv}")
        return None

    # Load results
    df = pd.read_csv(results_csv)

    # Clean column names (remove whitespace)
    df.columns = df.columns.str.strip()

    return df


def plot_loss_curves(df, output_dir):
    """
    Plot training and validation loss curves
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Loss Curves', fontsize=16, fontweight='bold')

    epochs = df['epoch'].values

    # Box loss
    axes[0, 0].plot(epochs, df['train/box_loss'], label='Train', linewidth=2, color='#2E86C1')
    axes[0, 0].plot(epochs, df['val/box_loss'], label='Validation', linewidth=2, color='#E74C3C')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Bounding Box Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Classification loss
    axes[0, 1].plot(epochs, df['train/cls_loss'], label='Train', linewidth=2, color='#2E86C1')
    axes[0, 1].plot(epochs, df['val/cls_loss'], label='Validation', linewidth=2, color='#E74C3C')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].set_title('Classification Loss', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # DFL loss (Distribution Focal Loss)
    axes[1, 0].plot(epochs, df['train/dfl_loss'], label='Train', linewidth=2, color='#2E86C1')
    axes[1, 0].plot(epochs, df['val/dfl_loss'], label='Validation', linewidth=2, color='#E74C3C')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].set_title('DFL Loss', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Total loss (combined)
    train_total = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
    val_total = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
    axes[1, 1].plot(epochs, train_total, label='Train', linewidth=2, color='#2E86C1')
    axes[1, 1].plot(epochs, val_total, label='Validation', linewidth=2, color='#E74C3C')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    axes[1, 1].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Loss curves saved: {output_path}")
    plt.close()


def plot_metrics_curves(df, output_dir):
    """
    Plot precision, recall, and mAP curves
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')

    epochs = df['epoch'].values

    # Precision
    axes[0, 0].plot(epochs, df['metrics/precision(B)'], linewidth=2, color='#27AE60')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Precision', fontsize=12)
    axes[0, 0].set_title('Precision', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylim([0, 1.05])
    axes[0, 0].grid(True, alpha=0.3)

    # Recall
    axes[0, 1].plot(epochs, df['metrics/recall(B)'], linewidth=2, color='#E67E22')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Recall', fontsize=12)
    axes[0, 1].set_title('Recall', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylim([0, 1.05])
    axes[0, 1].grid(True, alpha=0.3)

    # mAP@50
    axes[1, 0].plot(epochs, df['metrics/mAP50(B)'], linewidth=2, color='#8E44AD')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('mAP@50', fontsize=12)
    axes[1, 0].set_title('mAP@50 (IoU=0.50)', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim([0, 1.05])
    axes[1, 0].grid(True, alpha=0.3)

    # mAP@50-95
    axes[1, 1].plot(epochs, df['metrics/mAP50-95(B)'], linewidth=2, color='#C0392B')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('mAP@50-95', fontsize=12)
    axes[1, 1].set_title('mAP@50-95 (IoU=0.50:0.95)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylim([0, 1.05])
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'metrics_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics curves saved: {output_path}")
    plt.close()


def plot_learning_rate(df, output_dir):
    """
    Plot learning rate schedule
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = df['epoch'].values

    # Plot learning rates for different parameter groups if available
    lr_cols = [col for col in df.columns if 'lr/' in col]

    if lr_cols:
        for col in lr_cols:
            label = col.replace('lr/', '')
            ax.plot(epochs, df[col], linewidth=2, label=label)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'learning_rate.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Learning rate plot saved: {output_path}")
    plt.close()


def plot_combined_overview(df, output_dir):
    """
    Create a single combined overview plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Overview - License Plate Detection', fontsize=16, fontweight='bold')

    epochs = df['epoch'].values

    # Total loss
    train_total = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
    val_total = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
    axes[0, 0].plot(epochs, train_total, label='Train', linewidth=2, color='#2E86C1')
    axes[0, 0].plot(epochs, val_total, label='Validation', linewidth=2, color='#E74C3C')
    axes[0, 0].set_title('Total Loss', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Box loss
    axes[0, 1].plot(epochs, df['train/box_loss'], label='Train', linewidth=2, color='#2E86C1')
    axes[0, 1].plot(epochs, df['val/box_loss'], label='Validation', linewidth=2, color='#E74C3C')
    axes[0, 1].set_title('Box Loss', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Precision
    axes[0, 2].plot(epochs, df['metrics/precision(B)'], linewidth=2, color='#27AE60')
    axes[0, 2].set_title('Precision', fontweight='bold')
    axes[0, 2].set_ylim([0, 1.05])
    axes[0, 2].grid(True, alpha=0.3)

    # Recall
    axes[1, 0].plot(epochs, df['metrics/recall(B)'], linewidth=2, color='#E67E22')
    axes[1, 0].set_title('Recall', fontweight='bold')
    axes[1, 0].set_ylim([0, 1.05])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].grid(True, alpha=0.3)

    # mAP@50
    axes[1, 1].plot(epochs, df['metrics/mAP50(B)'], linewidth=2, color='#8E44AD')
    axes[1, 1].set_title('mAP@50', fontweight='bold')
    axes[1, 1].set_ylim([0, 1.05])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].grid(True, alpha=0.3)

    # mAP@50-95
    axes[1, 2].plot(epochs, df['metrics/mAP50-95(B)'], linewidth=2, color='#C0392B')
    axes[1, 2].set_title('mAP@50-95', fontweight='bold')
    axes[1, 2].set_ylim([0, 1.05])
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_overview.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training overview saved: {output_path}")
    plt.close()


def print_final_metrics(df):
    """
    Print final training metrics
    """
    last_epoch = df.iloc[-1]

    print(f"\n{'='*60}")
    print(f"FINAL TRAINING METRICS (Epoch {int(last_epoch['epoch'])})")
    print(f"{'='*60}")

    print(f"\nLoss Values:")
    print(f"  Train Box Loss: {last_epoch['train/box_loss']:.4f}")
    print(f"  Val Box Loss: {last_epoch['val/box_loss']:.4f}")
    print(f"  Train Cls Loss: {last_epoch['train/cls_loss']:.4f}")
    print(f"  Val Cls Loss: {last_epoch['val/cls_loss']:.4f}")
    print(f"  Train DFL Loss: {last_epoch['train/dfl_loss']:.4f}")
    print(f"  Val DFL Loss: {last_epoch['val/dfl_loss']:.4f}")

    print(f"\nMetrics:")
    print(f"  Precision: {last_epoch['metrics/precision(B)']:.4f}")
    print(f"  Recall: {last_epoch['metrics/recall(B)']:.4f}")
    print(f"  mAP@50: {last_epoch['metrics/mAP50(B)']:.4f}")
    print(f"  mAP@50-95: {last_epoch['metrics/mAP50-95(B)']:.4f}")

    # Find best metrics
    best_map50 = df['metrics/mAP50(B)'].max()
    best_map50_epoch = df['metrics/mAP50(B)'].idxmax()
    best_map50_95 = df['metrics/mAP50-95(B)'].max()
    best_map50_95_epoch = df['metrics/mAP50-95(B)'].idxmax()

    print(f"\nBest Metrics:")
    print(f"  Best mAP@50: {best_map50:.4f} (Epoch {int(df.iloc[best_map50_epoch]['epoch'])})")
    print(f"  Best mAP@50-95: {best_map50_95:.4f} (Epoch {int(df.iloc[best_map50_95_epoch]['epoch'])})")

    print(f"\n{'='*60}\n")


def main():
    """
    Main visualization function
    """
    # Default path (can be modified)
    RESULTS_DIR = r"C:\Users\SelmaB\Desktop\detection\runs\detect\license_plate_detection4"

    print(f"\n{'='*60}")
    print(f"TRAINING RESULTS VISUALIZATION")
    print(f"{'='*60}")
    print(f"Results directory: {RESULTS_DIR}\n")

    # Check if directory exists
    if not os.path.exists(RESULTS_DIR):
        print(f"ERROR: Results directory not found!")
        print(f"Please update RESULTS_DIR in the script or train the model first.")

        # Try to find the latest run
        runs_dir = r"C:\Users\SelmaB\Desktop\detection\runs\detect"
        if os.path.exists(runs_dir):
            runs = sorted(Path(runs_dir).glob('license_plate_detection*'))
            if runs:
                latest_run = runs[-1]
                print(f"\nFound latest run: {latest_run}")
                print(f"Update RESULTS_DIR to: {latest_run}")
        return

    # Load results
    print("Loading training results...")
    df = load_training_results(RESULTS_DIR)

    if df is None:
        return

    print(f"✓ Loaded {len(df)} epochs of training data")

    # Create visualizations directory
    viz_dir = os.path.join(RESULTS_DIR, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    print(f"\nGenerating visualizations in: {viz_dir}\n")

    # Generate plots
    print("Generating plots...")
    plot_loss_curves(df, viz_dir)
    plot_metrics_curves(df, viz_dir)
    plot_learning_rate(df, viz_dir)
    plot_combined_overview(df, viz_dir)

    # Print final metrics
    print_final_metrics(df)

    print(f"✓ All visualizations generated successfully!")
    print(f"\nPlots saved in: {viz_dir}")


if __name__ == "__main__":
    main()