"""
Quick Example: Test Character-Level Metrics
============================================

Simple example showing how the metrics work with sample data.
Run this to understand the metrics before evaluating your real data.
"""

from ocr_evaluation import (
    character_accuracy,
    edit_distance_accuracy,
    exact_match_accuracy,
    normalize_plate_text,
    compare_preprocessing_impact
)


def print_separator(title=""):
    """Print a nice separator."""
    print("\n" + "="*70)
    if title:
        print(f"  {title}")
        print("="*70)
    print()


def example_1_single_prediction():
    """Example 1: Compare a single prediction to ground truth."""
    print_separator("EXAMPLE 1: Single Prediction Evaluation")

    ground_truth = "123 تونس 456"
    prediction = "123TN445"  # Almost correct - one digit wrong

    print(f"Ground Truth:  {ground_truth}")
    print(f"Prediction:    {prediction}")
    print(f"\nNormalized GT:   {normalize_plate_text(ground_truth)}")
    print(f"Normalized Pred: {normalize_plate_text(prediction)}")

    char_acc = character_accuracy(ground_truth, prediction)
    edit_acc = edit_distance_accuracy(ground_truth, prediction)
    exact = exact_match_accuracy(ground_truth, prediction)

    print(f"\n{'Metric':<30} {'Score':>10}")
    print("-"*40)
    print(f"{'Exact Match':<30} {'Yes' if exact else 'No':>10}")
    print(f"{'Character-Level Accuracy':<30} {char_acc*100:>9.1f}%")
    print(f"{'Edit Distance Accuracy':<30} {edit_acc*100:>9.1f}%")

    print("\n💡 Interpretation:")
    print("   The prediction is 87.5% correct (7 out of 8 characters)")
    print("   Only 1 character needs to be fixed")


def example_2_comparison():
    """Example 2: Compare two predictions."""
    print_separator("EXAMPLE 2: Comparing Two Predictions")

    ground_truth = "123 تونس 456"
    prediction_a = "123TN445"  # One error
    prediction_b = "999TN999"  # Multiple errors

    print(f"Ground Truth:  {ground_truth}")
    print(f"Prediction A:  {prediction_a}  (almost correct)")
    print(f"Prediction B:  {prediction_b}  (many errors)")

    # Calculate metrics for both
    char_acc_a = character_accuracy(ground_truth, prediction_a)
    char_acc_b = character_accuracy(ground_truth, prediction_b)

    edit_acc_a = edit_distance_accuracy(ground_truth, prediction_a)
    edit_acc_b = edit_distance_accuracy(ground_truth, prediction_b)

    print(f"\n{'Metric':<30} {'Pred A':>12} {'Pred B':>12}")
    print("-"*55)
    print(f"{'Character Accuracy':<30} {char_acc_a*100:>11.1f}% {char_acc_b*100:>11.1f}%")
    print(f"{'Edit Distance Accuracy':<30} {edit_acc_a*100:>11.1f}% {edit_acc_b*100:>11.1f}%")

    print("\n💡 Interpretation:")
    print(f"   Prediction A is clearly better ({char_acc_a*100:.1f}% vs {char_acc_b*100:.1f}%)")
    print("   Character-level metrics capture this difference!")


def example_3_preprocessing_impact():
    """Example 3: Evaluate preprocessing impact on multiple images."""
    print_separator("EXAMPLE 3: Preprocessing Impact Evaluation")

    # Simulate test data (5 images)
    ground_truths = [
        "123 تونس 456",
        "789 تونس 012",
        "111 تونس 222",
        "333 تونس 444",
        "555 تونس 666"
    ]

    # Predictions WITHOUT preprocessing (some errors)
    predictions_without = [
        "12345",        # Missing part
        "78901",        # Missing part
        "111TN22",      # One digit missing
        "333444",       # Missing "تونس"
        "555TN666"      # Perfect
    ]

    # Predictions WITH preprocessing (better)
    predictions_with = [
        "123TN456",     # Perfect
        "789TN012",     # Perfect
        "111TN222",     # Perfect
        "333TN444",     # Perfect
        "555TN666"      # Perfect
    ]

    print("Simulating OCR on 5 test images...")
    print("\nTest Set:")
    for i, gt in enumerate(ground_truths):
        print(f"  {i+1}. {gt}")

    # Run comparison
    comparison = compare_preprocessing_impact(
        ground_truths,
        predictions_without,
        predictions_with,
        verbose=False
    )

    # Print results
    without = comparison['without_preprocessing']
    with_preproc = comparison['with_preprocessing']
    improvements = comparison['improvements']

    print(f"\n{'Metric':<30} {'Without':>12} {'With':>12} {'Improvement':>12}")
    print("-"*70)

    print(f"{'Exact Match Accuracy':<30} "
          f"{without['exact_match_accuracy']*100:>11.1f}% "
          f"{with_preproc['exact_match_accuracy']*100:>11.1f}% "
          f"{improvements['exact_match_improvement']*100:>+11.1f}%")

    print(f"{'Character Accuracy':<30} "
          f"{without['avg_char_accuracy']*100:>11.1f}% "
          f"{with_preproc['avg_char_accuracy']*100:>11.1f}% "
          f"{improvements['char_accuracy_improvement']*100:>+11.1f}%")

    print(f"{'Edit Distance Accuracy':<30} "
          f"{without['avg_edit_accuracy']*100:>11.1f}% "
          f"{with_preproc['avg_edit_accuracy']*100:>11.1f}% "
          f"{improvements['edit_accuracy_improvement']*100:>+11.1f}%")

    print("\n💡 Interpretation:")
    if improvements['char_accuracy_improvement'] > 0:
        print(f"   ✓ Preprocessing improves accuracy by {improvements['char_accuracy_improvement']*100:.1f}%")
        print("   This shows preprocessing makes OCR more accurate!")
    else:
        print("   Preprocessing doesn't improve accuracy in this example")

    # Show per-sample details
    print("\n\nPer-Sample Results:")
    print(f"{'GT':<12} {'Without':<15} {'With':<15} {'Status':<10}")
    print("-"*70)

    for i in range(len(without['per_sample_results'])):
        w = without['per_sample_results'][i]
        p = with_preproc['per_sample_results'][i]

        gt = w['normalized_gt']
        pred_w = w['normalized_pred']
        pred_p = p['normalized_pred']

        if p['exact_match']:
            status = "✓ Perfect"
        elif p['char_accuracy'] > w['char_accuracy']:
            status = "↑ Better"
        else:
            status = "= Same"

        print(f"{gt:<12} {pred_w:<15} {pred_p:<15} {status:<10}")


def example_4_edge_cases():
    """Example 4: Edge cases and special scenarios."""
    print_separator("EXAMPLE 4: Edge Cases")

    test_cases = [
        ("123 تونس 456", "123 تونس 456", "Perfect match"),
        ("123 تونس 456", "", "Empty prediction"),
        ("123 تونس 456", "999999999", "Completely wrong"),
        ("123 تونس 456", "456 تونس 123", "Reversed numbers"),
        ("123 تونس 456", "123456", "Missing تونس"),
    ]

    print(f"{'Ground Truth':<20} {'Prediction':<20} {'Char Acc':>10} {'Edit Acc':>10} {'Case':<20}")
    print("-"*90)

    for gt, pred, case in test_cases:
        char_acc = character_accuracy(gt, pred)
        edit_acc = edit_distance_accuracy(gt, pred)

        print(f"{gt:<20} {pred:<20} {char_acc*100:>9.1f}% {edit_acc*100:>9.1f}% {case:<20}")

    print("\n💡 Interpretation:")
    print("   The metrics handle various edge cases gracefully")
    print("   Empty predictions → 0% accuracy")
    print("   Perfect matches → 100% accuracy")
    print("   Partial matches → proportional scores")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("  CHARACTER-LEVEL OCR EVALUATION - EXAMPLES")
    print("="*70)

    example_1_single_prediction()
    input("\nPress Enter to continue to Example 2...")

    example_2_comparison()
    input("\nPress Enter to continue to Example 3...")

    example_3_preprocessing_impact()
    input("\nPress Enter to continue to Example 4...")

    example_4_edge_cases()

    print("\n" + "="*70)
    print("  EXAMPLES COMPLETE")
    print("="*70)
    print("\n✓ Now you understand how the metrics work!")
    print("\nNext steps:")
    print("  1. Run test_preprocessing_impact.py on your real data")
    print("  2. Or use: streamlit run app_ocr_evaluation.py")
    print("  3. Read EVALUATION_GUIDE.md for full documentation")
    print()


if __name__ == "__main__":
    main()