"""
Visualize Test Results
Generates comprehensive plots for test set evaluation
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json


def find_test_results_dir():
    """
    Find the test results directory
    """
    # Look for validation results in runs/detect/
    runs_dir = Path("runs/detect")

    if not runs_dir.exists():
        print(f"ERROR: runs/detect directory not found!")
        return None

    # Look for directories with 'val' in the name (created by model.val())
    val_dirs = sorted(runs_dir.glob('val*'))

    if not val_dirs:
        print(f"ERROR: No validation/test results found in runs/detect/")
        print(f"Please run test_model.py first!")
        return None

    # Get the most recent one
    latest_val = val_dirs[-1]
    print(f"Found test results: {latest_val}")

    return latest_val


def load_test_metrics(results_dir):
    """
    Load metrics from test results directory
    """
    # Try to find results files
    csv_file = results_dir / 'results.csv'

    if csv_file.exists():
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
        return df

    return None


def plot_confusion_matrix(results_dir, output_dir):
    """
    Display confusion matrix if available
    """
    cm_file = results_dir / 'confusion_matrix.png'
    cm_normalized_file = results_dir / 'confusion_matrix_normalized.png'

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Test Set - Confusion Matrices', fontsize=16, fontweight='bold')

    found_plots = False

    if cm_file.exists():
        img = plt.imread(cm_file)
        axes[0].imshow(img)
        axes[0].axis('off')
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        found_plots = True
    else:
        axes[0].text(0.5, 0.5, 'Confusion Matrix\nNot Available',
                    ha='center', va='center', fontsize=14, color='gray')
        axes[0].axis('off')

    if cm_normalized_file.exists():
        img = plt.imread(cm_normalized_file)
        axes[1].imshow(img)
        axes[1].axis('off')
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        found_plots = True
    else:
        axes[1].text(0.5, 0.5, 'Normalized Confusion Matrix\nNot Available',
                    ha='center', va='center', fontsize=14, color='gray')
        axes[1].axis('off')

    plt.tight_layout()
    output_path = output_dir / 'test_confusion_matrices.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrices saved: {output_path}")
    plt.close()

    return found_plots


def plot_pr_curve(results_dir, output_dir):
    """
    Display Precision-Recall curve if available
    """
    pr_curve_file = results_dir / 'PR_curve.png'

    if pr_curve_file.exists():
        fig, ax = plt.subplots(figsize=(10, 8))
        img = plt.imread(pr_curve_file)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title('Test Set - Precision-Recall Curve', fontsize=16, fontweight='bold')

        plt.tight_layout()
        output_path = output_dir / 'test_pr_curve.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ PR curve saved: {output_path}")
        plt.close()
        return True

    return False


def plot_f1_curve(results_dir, output_dir):
    """
    Display F1-Confidence curve if available
    """
    f1_curve_file = results_dir / 'F1_curve.png'

    if f1_curve_file.exists():
        fig, ax = plt.subplots(figsize=(10, 8))
        img = plt.imread(f1_curve_file)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title('Test Set - F1-Confidence Curve', fontsize=16, fontweight='bold')

        plt.tight_layout()
        output_path = output_dir / 'test_f1_curve.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ F1 curve saved: {output_path}")
        plt.close()
        return True

    return False


def create_metrics_comparison(output_dir):
    """
    Create a comparison chart between validation and test metrics
    """
    # Validation metrics (from training)
    val_metrics = {
        'Precision': 0.999,
        'Recall': 0.962,
        'mAP@50': 0.988,
        'mAP@50-95': 0.771
    }

    # Test metrics (you'll need to update these after running test)
    # For now, we'll create a template
    fig, ax = plt.subplots(figsize=(12, 7))

    metrics = list(val_metrics.keys())
    val_values = list(val_metrics.values())

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, val_values, width, label='Validation Set',
                   color='#2E86C1', alpha=0.8)

    # Placeholder for test values (will be filled after test)
    test_values = [0.95, 0.93, 0.96, 0.75]  # Placeholder values
    bars2 = ax.bar(x + width/2, test_values, width, label='Test Set (Run test_model.py)',
                   color='#E74C3C', alpha=0.5, hatch='//')

    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Validation vs Test Set Performance\n(Update after running test_model.py)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / 'validation_vs_test_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison chart saved: {output_path}")
    plt.close()


def create_test_summary_visualization(results_dir, output_dir):
    """
    Create a comprehensive summary visualization
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Test Set Evaluation - Comprehensive Summary',
                 fontsize=18, fontweight='bold', y=0.98)

    # Try to load various plots
    plots = {
        'Confusion Matrix': results_dir / 'confusion_matrix.png',
        'PR Curve': results_dir / 'PR_curve.png',
        'F1 Curve': results_dir / 'F1_curve.png',
        'P Curve': results_dir / 'P_curve.png',
        'R Curve': results_dir / 'R_curve.png',
    }

    # Create subplots
    positions = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
    ]

    plot_count = 0
    for (name, path), (row, col) in zip(plots.items(), positions):
        ax = fig.add_subplot(gs[row, col])

        if path.exists():
            img = plt.imread(path)
            ax.imshow(img)
            ax.set_title(name, fontsize=12, fontweight='bold')
            plot_count += 1
        else:
            ax.text(0.5, 0.5, f'{name}\nNot Available',
                   ha='center', va='center', fontsize=11, color='gray')

        ax.axis('off')

    # Add text summary in bottom row
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis('off')

    summary_text = """
    TEST SET EVALUATION SUMMARY

    This visualization shows the model's performance on the held-out test set (135 images).

    Key Points:
    • Test set was NOT used during training or validation
    • These metrics represent true, unbiased performance
    • Compare with validation metrics to check for overfitting

    To run the test evaluation:
        python test_model.py  (or run RUN_TEST.bat)

    Generated plots include:
    • Confusion Matrix: Shows classification accuracy
    • PR Curve: Precision vs Recall trade-off
    • F1 Curve: F1 score across confidence thresholds
    • P/R Curves: Precision and Recall vs confidence
    """

    ax_text.text(0.5, 0.5, summary_text,
                ha='center', va='center',
                fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    output_path = output_dir / 'test_summary_comprehensive.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comprehensive summary saved: {output_path}")
    plt.close()

    return plot_count > 0


def plot_sample_predictions(results_dir, output_dir):
    """
    Display sample predictions from test set if available
    """
    # Look for prediction images
    pred_dir = results_dir / 'labels'

    if not pred_dir.exists():
        pred_dir = results_dir

    # Find sample images with predictions
    sample_images = list(pred_dir.glob('val_batch*.jpg'))

    if not sample_images:
        print("⚠ No sample prediction images found")
        return False

    # Display first few batches
    num_samples = min(3, len(sample_images))

    fig, axes = plt.subplots(1, num_samples, figsize=(18, 6))
    if num_samples == 1:
        axes = [axes]

    fig.suptitle('Test Set - Sample Predictions', fontsize=16, fontweight='bold')

    for i, img_path in enumerate(sample_images[:num_samples]):
        img = plt.imread(img_path)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Batch {i+1}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / 'test_sample_predictions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Sample predictions saved: {output_path}")
    plt.close()

    return True


def main():
    """
    Main visualization function
    """
    print(f"\n{'='*70}")
    print(f"  TEST RESULTS VISUALIZATION")
    print(f"{'='*70}\n")

    # Find test results directory
    results_dir = find_test_results_dir()

    if results_dir is None:
        print("\n⚠ No test results found!")
        print("\nPlease run the test first:")
        print("  python test_model.py")
        print("  (or run RUN_TEST.bat)")
        return

    print(f"Loading test results from: {results_dir}\n")

    # Create output directory for visualizations
    output_dir = results_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    print(f"Saving visualizations to: {output_dir}\n")

    print("Generating visualizations...\n")

    # Generate all visualizations
    plots_generated = 0

    # 1. Confusion matrices
    if plot_confusion_matrix(results_dir, output_dir):
        plots_generated += 1

    # 2. PR curve
    if plot_pr_curve(results_dir, output_dir):
        plots_generated += 1

    # 3. F1 curve
    if plot_f1_curve(results_dir, output_dir):
        plots_generated += 1

    # 4. Comparison chart
    create_metrics_comparison(output_dir)
    plots_generated += 1

    # 5. Comprehensive summary
    if create_test_summary_visualization(results_dir, output_dir):
        plots_generated += 1

    # 6. Sample predictions
    if plot_sample_predictions(results_dir, output_dir):
        plots_generated += 1

    print(f"\n{'='*70}")
    print(f"  VISUALIZATION COMPLETE")
    print(f"{'='*70}\n")
    print(f"✓ Generated {plots_generated} visualization(s)")
    print(f"✓ Saved to: {output_dir}")
    print(f"\nGenerated files:")
    for file in sorted(output_dir.glob('*.png')):
        print(f"  - {file.name}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
