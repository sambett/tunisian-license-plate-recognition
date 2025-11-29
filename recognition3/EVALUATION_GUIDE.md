# OCR Evaluation Guide

## Character-Level Metrics for License Plate Recognition

This guide explains how to evaluate OCR performance using **character-level accuracy** and **edit distance** instead of strict all-or-nothing plate matching.

---

## Why Character-Level Metrics?

Traditional OCR evaluation treats "123445" and "999999" the same way when the ground truth is "123456" - both are **wrong**.

But intuitively:
- **"123445"** is much better - it has 5 out of 6 characters correct (83.3%)
- **"999999"** is completely wrong - 0% correct

Character-level metrics capture this difference, giving you:
1. **Partial credit** for almost-correct predictions
2. **Better insight** into preprocessing effectiveness
3. **More realistic** evaluation for OCR systems

---

## Metrics Explained

### 1. Character-Level Accuracy

**What it measures:** Percentage of correctly recognized characters at each position.

**Formula:**
```
CharAcc = (# correct characters) / (# total characters)
```

**Example:**
- Ground truth: `123 تونس 456` → normalized to `123TN456`
- Prediction: `123TN445`
- Result: 7 out of 8 correct → **87.5% accuracy**

**Interpretation:**
- 100% = Perfect match
- 80-99% = Almost correct (1-2 character errors)
- 50-79% = Partially correct
- <50% = Poor recognition

---

### 2. Edit Distance Accuracy

**What it measures:** How many edits (insertions, deletions, substitutions) are needed to fix the prediction.

**Formula:**
```
EditAcc = 1 - (Levenshtein distance / max length)
```

**Example:**
- Ground truth: `123456`
- Prediction: `123445` → distance = 1 (one substitution)
- Result: `1 - 1/6 = 0.833` → **83.3% accuracy**

**Interpretation:**
- 100% = Perfect match
- 80-99% = Very close (1-2 edits needed)
- 60-79% = Several errors
- <60% = Very different from ground truth

---

### 3. Exact Match Accuracy (for reference)

**What it measures:** Traditional strict accuracy - plate must be 100% correct.

**Formula:**
```
ExactMatch = (# perfectly correct plates) / (# total plates)
```

**Use case:** Good for final deployment metrics, but too strict for evaluating preprocessing impact.

---

## How to Use the Evaluation Tools

### Method 1: Command-Line Evaluation

Use `test_preprocessing_impact.py` to evaluate on your dataset:

```bash
cd recognition3
python test_preprocessing_impact.py
```

**Configuration:**
Edit the script to point to your test data:

```python
# Use existing dataset
USE_EXISTING_DATASET = True
EXISTING_CSV = "license_plates_recognition_train.csv"
EXISTING_DIR = "license_plates_recognition_train"
TEST_SET_SIZE = 20  # Number of images to test

# Or create custom test set
USE_EXISTING_DATASET = False
TEST_ANNOTATIONS_CSV = "test_annotations.csv"
TEST_IMAGES_DIR = "test_images"
```

**Output:**
- Console: Detailed comparison report
- CSV file: Per-image results with all metrics

---

### Method 2: Interactive Dashboard

Use `app_ocr_evaluation.py` for interactive evaluation:

```bash
cd recognition3
streamlit run app_ocr_evaluation.py
```

**Features:**
- Upload test images with ground truth
- Run OCR with/without preprocessing
- View interactive charts
- Download results as CSV

**Input Methods:**
1. **Upload Images**: Upload images + enter ground truth for each
2. **Manual Entry**: Type predictions manually (for quick tests)
3. **CSV Upload**: Upload CSV with columns: `ground_truth`, `pred_without`, `pred_with`

---

### Method 3: Python API

Use the evaluation functions in your own scripts:

```python
from ocr_evaluation import (
    character_accuracy,
    edit_distance_accuracy,
    compare_preprocessing_impact
)

# Example: Compare single prediction
gt = "123 تونس 456"
pred = "123TN445"

char_acc = character_accuracy(gt, pred)
edit_acc = edit_distance_accuracy(gt, pred)

print(f"Character accuracy: {char_acc*100:.1f}%")
print(f"Edit distance accuracy: {edit_acc*100:.1f}%")

# Example: Batch comparison
ground_truths = ["123 تونس 456", "789 تونس 012"]
predictions_without = ["12345", "78901"]
predictions_with = ["123TN456", "789TN012"]

comparison = compare_preprocessing_impact(
    ground_truths,
    predictions_without,
    predictions_with
)
```

---

## Creating Test Annotations

### Option 1: CSV Format

Create `test_annotations.csv`:

```csv
img_id,text
001.jpg,123 تونس 456
002.jpg,789 تونس 012
003.jpg,111 تونس 222
```

### Option 2: Use Existing Dataset

If you already have annotated data (like your training set), just use it:

```python
USE_EXISTING_DATASET = True
EXISTING_CSV = "license_plates_recognition_train.csv"
TEST_SET_SIZE = 20  # Sample 20 images for evaluation
```

---

## Interpreting Results

### Sample Output

```
================================================================================
COMPARISON RESULTS
================================================================================

Metric                              Without Preproc    With Preproc       Improvement
--------------------------------------------------------------------------------
Exact Match Accuracy                 15.00% (3/20)      65.00% (13/20)      +50.00%
Character-Level Accuracy             67.50%             89.20%              +21.70%
Edit Distance Accuracy               64.30%             87.80%              +23.50%

================================================================================
```

### What This Tells You

1. **Exact Match** improved from 15% to 65% → Preprocessing helps significantly

2. **Character Accuracy** improved from 67.5% to 89.2% → Even when plates aren't perfect, preprocessing gets more characters correct

3. **Edit Distance** improved from 64.3% to 87.8% → Predictions are much closer to ground truth

### Key Insight

Even when **exact match** is low (15%), the **character accuracy** might be reasonable (67.5%), meaning:
- The OCR is getting most characters right
- Just 1-2 character errors per plate
- Small improvements in preprocessing could push accuracy much higher

---

## Use in Your Report

### Evaluation Protocol Section

You can copy this into your report:

> **OCR Evaluation Metrics**
>
> To evaluate the impact of plate preprocessing on OCR performance, we built a test set of 20 annotated images with ground-truth plate text. We ran the full pipeline in two configurations: (1) without preprocessing (raw crop → OCR) and (2) with our enhancement pipeline enabled (crop → preprocessing → OCR).
>
> Instead of using a binary "correct/incorrect plate" metric, we focused on **character-level evaluation**. For each image, we compared the OCR prediction with the ground truth after normalization and computed:
>
> 1. **Character-level accuracy**: Ratio of correctly recognized characters at each position. This gives partial credit when most characters are correct, even if the whole plate is not perfect.
>
> 2. **Normalized edit-distance accuracy**: Based on the Levenshtein distance, which counts the minimum number of insertions, deletions, and substitutions needed to transform the prediction into the ground truth. Converted to a score between 0 and 1:
>    ```
>    EditAcc = 1 - (edit_distance / max_length)
>    ```
>
> 3. **Exact match accuracy**: Traditional strict metric where the plate must be 100% correct (for reference).
>
> We then averaged these scores over all images. This approach allows us to distinguish predictions that are "almost correct" (most digits right) from those that are "completely wrong", and to quantify how much the preprocessing stage improves the quality of the recognized plate text.

### Results Section

Example results text:

> **Preprocessing Impact on OCR Performance**
>
> Our evaluation on 20 test images showed that plate preprocessing significantly improved OCR accuracy across all metrics:
>
> - **Exact match accuracy** improved from 15.0% to 65.0% (+50.0 percentage points)
> - **Character-level accuracy** improved from 67.5% to 89.2% (+21.7 pp)
> - **Edit distance accuracy** improved from 64.3% to 87.8% (+23.5 pp)
>
> The character-level metrics reveal that even without preprocessing, the OCR gets approximately two-thirds of characters correct on average. However, preprocessing brings this to nearly 90%, reducing character-level errors by more than two-thirds. This demonstrates that the enhancement pipeline (denoising, contrast adjustment, and binarization) substantially improves the readability of license plate images for the OCR engine.

---

## Advanced Usage

### Custom Normalization

Modify `normalize_plate_text()` in [ocr_evaluation.py](ocr_evaluation.py:37) for different plate formats:

```python
def normalize_plate_text(text: str) -> str:
    # Remove spaces
    text = text.replace(" ", "").replace("-", "")

    # Handle Arabic text
    text = text.replace("تونس", "TN")

    # Add custom rules for your format
    # ...

    return text.upper()
```

### Additional Metrics

Add custom metrics to the evaluation:

```python
def custom_metric(gt, pred):
    # Your metric here
    return score

# In evaluate_ocr_predictions()
custom_score = custom_metric(gt, pred)
results['custom_scores'].append(custom_score)
```

---

## Files Overview

| File | Purpose |
|------|---------|
| `ocr_evaluation.py` | Core evaluation functions and metrics |
| `test_preprocessing_impact.py` | Command-line evaluation script |
| `app_ocr_evaluation.py` | Interactive Streamlit dashboard |
| `EVALUATION_GUIDE.md` | This documentation |

---

## Troubleshooting

### Issue: "No test data loaded"

**Solution:** Make sure your CSV file exists and has the correct columns:
- `img_id` or `image`: Image filename
- `text` or `ground_truth`: Ground truth plate text

### Issue: OCR returns empty strings

**Solution:** Check that:
1. EasyOCR is installed: `pip install easyocr`
2. Images are readable: `cv2.imread(image_path)` works
3. Images contain visible text (not too dark/blurry)

### Issue: Metrics are all 0%

**Solution:**
1. Check text normalization is working correctly
2. Verify ground truth format matches OCR output format
3. Print raw predictions to debug

---

## Citation

If you use these evaluation methods in your report, you can reference the Levenshtein distance as:

> Levenshtein, V. I. (1966). Binary codes capable of correcting deletions, insertions, and reversals. *Soviet Physics Doklady*, 10(8), 707-710.

For character-level accuracy in OCR, cite:

> Graves, A., et al. (2009). A novel connectionist system for unconstrained handwriting recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 31(5), 855-868.

---

## Questions?

For issues or questions:
1. Check this guide
2. Review code comments in `ocr_evaluation.py`
3. Test with the example data provided in `__main__` blocks

---

**Good luck with your evaluation!** 🚀