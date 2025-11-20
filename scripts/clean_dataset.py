"""
===================================================================
DATASET CLEANING SCRIPT FOR YOLO FORMAT
===================================================================

PURPOSE:
    Permanently DELETE images that don't have corresponding annotation (.txt) files
    This ensures YOLOv8 training won't encounter errors from missing labels

⚠️  WARNING - PERMANENT DELETION:
    - This script PERMANENTLY DELETES files (no backup!)
    - Deleted files CANNOT be recovered
    - Always shows a dry run preview before deleting
    - Requires explicit user confirmation

WHAT IT DOES:
    1. Scans your dataset for images without matching .txt label files
    2. Shows you exactly what will be DELETED (dry run preview)
    3. Asks for your confirmation
    4. PERMANENTLY DELETES unpaired images
    5. Shows final statistics

EXAMPLE:
    If you have:
        - images/train/car_001.jpg (has car_001.txt) ✅ KEEP
        - images/train/car_002.jpg (NO car_002.txt)  ❌ DELETE
        - images/train/car_003.jpg (has car_003.txt) ✅ KEEP

    Result:
        - car_002.jpg is PERMANENTLY DELETED
        - Only car_001.jpg and car_003.jpg remain
===================================================================
"""

import os
from pathlib import Path
from typing import Tuple, List


class DatasetCleaner:
    """
    A class to clean YOLO format datasets by DELETING unpaired images

    YOLO format requires:
        - images/train/image_name.jpg
        - labels/train/image_name.txt

    This class finds and PERMANENTLY DELETES images without matching .txt files
    """

    def __init__(self, dataset_root: str):
        """
        Initialize the dataset cleaner

        Args:
            dataset_root: Path to dataset root folder that contains:
                         - images/ folder (with train/ and test/ subfolders)
                         - labels/ folder (with train/ and test/ subfolders)

        Example:
            dataset_root = "dataset/content/UA-DETRAC/DETRAC_Upload"
            Structure:
                dataset_root/
                ├── images/
                │   ├── train/  (contains .jpg files)
                │   └── test/   (contains .jpg files)
                └── labels/
                    ├── train/  (contains .txt files)
                    └── test/   (contains .txt files)
        """
        self.dataset_root = Path(dataset_root)
        self.images_dir = self.dataset_root / "images"
        self.labels_dir = self.dataset_root / "labels"

    def find_unpaired_images(self, split: str = "train") -> List[Path]:
        """
        Find all images that don't have corresponding label files

        HOW IT WORKS:
            1. Get list of ALL image files in the split folder
            2. For EACH image file (e.g., car_001.jpg):
               - Look for matching label file (car_001.txt) in labels folder
               - If .txt file doesn't exist → add image to unpaired list
            3. Return complete list of all unpaired images

        WHY WE DO THIS:
            YOLOv8 expects every image to have a matching .txt label file
            Images without labels will cause training errors or be skipped
            We identify these to clean them out

        Args:
            split: Dataset split to check ('train' or 'test')

        Returns:
            List of Path objects for images without labels

        Example:
            Input:
                images/train/ has 100 images
                labels/train/ has 98 .txt files
            Output:
                List containing 2 image paths that don't have .txt files
                These 2 images will be DELETED when cleaning runs
        """
        # Build paths to images and labels folders
        images_path = self.images_dir / split  # e.g., dataset/images/train
        labels_path = self.labels_dir / split  # e.g., dataset/labels/train

        # Safety check: make sure folders exist before proceeding
        if not images_path.exists():
            raise ValueError(f"Images directory not found: {images_path}")
        if not labels_path.exists():
            raise ValueError(f"Labels directory not found: {labels_path}")

        # Define supported image formats
        # We check for .jpg, .jpeg, .png, and .bmp files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        # Collect all image files from the folder
        all_images = []
        for ext in image_extensions:
            # glob finds all files matching pattern (e.g., "*.jpg")
            # We extend (not append) to flatten the list
            all_images.extend(images_path.glob(f"*{ext}"))

        # List to store images without matching labels
        # These are the images that will be DELETED
        unpaired_images = []

        print(f"\n[{split.upper()}] Checking {len(all_images)} images...")

        # Check each image for a corresponding label file
        for img_path in all_images:
            # Construct expected label path
            # Example:
            #   Image: car_001.jpg → Label: car_001.txt
            #   Image: MVI_20011_img00230.jpg → Label: MVI_20011_img00230.txt
            # img_path.stem extracts filename without extension
            label_path = labels_path / (img_path.stem + '.txt')

            # Check if label file exists
            if not label_path.exists():
                # No label found! This image is unpaired
                # Add it to the deletion list
                unpaired_images.append(img_path)

        return unpaired_images

    def clean_split(self, split: str = "train", dry_run: bool = True) -> Tuple[int, int]:
        """
        Clean a dataset split by PERMANENTLY DELETING unpaired images

        ⚠️  WARNING: This function PERMANENTLY DELETES files!

        PROCESS:
            1. Count total images in the split (before cleaning)
            2. Find all unpaired images (images without .txt label files)
            3. If dry_run=True:
               - Just SHOW what would be deleted
               - NO files are actually deleted
               - This is a PREVIEW/SAFETY CHECK
            4. If dry_run=False:
               - Actually DELETE the unpaired images
               - This is PERMANENT and IRREVERSIBLE

        IMPORTANT:
            - Deleted files CANNOT be recovered
            - Always run with dry_run=True first to preview
            - Only set dry_run=False after reviewing the preview

        Args:
            split: Dataset split ('train' or 'test')
            dry_run: If True, only SHOW what would be deleted (no actual deletion)
                    If False, PERMANENTLY DELETE the unpaired images

        Returns:
            Tuple of (total_images_before_cleaning, number_of_images_deleted)

        Example:
            # Step 1: Preview what would be deleted (SAFE)
            clean_split("train", dry_run=True)
            → Shows: "Would delete 1,706 images"
            → Does: NOTHING - just shows a preview

            # Step 2: Actually delete after confirming preview (DANGER!)
            clean_split("train", dry_run=False)
            → Deletes: 1,706 images PERMANENTLY
            → Does: Actually deletes the files (CANNOT be undone!)
        """
        images_path = self.images_dir / split

        # ========================================
        # STEP 1: Count total images before cleaning
        # ========================================
        # We count images to show statistics (before/after comparison)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        total_images = sum(len(list(images_path.glob(f"*{ext}")))
                          for ext in image_extensions)

        # ========================================
        # STEP 2: Find unpaired images
        # ========================================
        # Call find_unpaired_images() to get list of images without labels
        # These are the images that will be DELETED
        unpaired_images = self.find_unpaired_images(split)

        # If no unpaired images found, dataset is already clean!
        if len(unpaired_images) == 0:
            print(f"✅ [{split.upper()}] All images have corresponding labels!")
            print(f"   No files need to be deleted - dataset is clean!")
            return total_images, 0

        # Show how many unpaired images were found
        print(f"\n⚠️  [{split.upper()}] Found {len(unpaired_images)} unpaired images")
        print(f"   These images will be PERMANENTLY DELETED")

        # ========================================
        # STEP 3: DRY RUN MODE (preview only - SAFE)
        # ========================================
        if dry_run:
            # DRY RUN = Just show what WOULD be deleted, don't actually delete anything
            # This is a SAFETY FEATURE to let you review before permanent deletion
            print(f"\n[DRY RUN] Would PERMANENTLY DELETE {len(unpaired_images)} images:")

            # Show first 10 files as preview so you can see what would be deleted
            for i, img in enumerate(unpaired_images[:10], 1):
                print(f"  {i}. {img.name}")

            # If there are more than 10, show count of remaining files
            if len(unpaired_images) > 10:
                print(f"  ... and {len(unpaired_images) - 10} more files")

            print(f"\n💡 This is a DRY RUN - no files were actually deleted")
            print(f"   Set dry_run=False to actually delete these files")

            return total_images, len(unpaired_images)

        # ========================================
        # STEP 4: ACTUAL DELETION (if not dry run - DANGER!)
        # ========================================
        print(f"\n🗑️  DELETING {len(unpaired_images)} unpaired images...")
        print(f"⚠️  WARNING: This is PERMANENT and CANNOT be undone!")

        # Counter to track how many files were successfully deleted
        deleted_count = 0

        # Delete each unpaired image
        for img_path in unpaired_images:
            try:
                # PERMANENTLY DELETE the file
                # unlink() is Python's way to delete a file
                img_path.unlink()
                deleted_count += 1

                # Optional: Show progress every 100 files for large deletions
                if deleted_count % 100 == 0:
                    print(f"   Progress: {deleted_count}/{len(unpaired_images)} deleted...")

            except Exception as e:
                # If deletion fails (file locked, permission error, etc.)
                # Show error but continue with other files
                print(f"❌ Error deleting {img_path.name}: {e}")

        # Show completion message
        print(f"\n✅ [{split.upper()}] Successfully DELETED {deleted_count} images")
        print(f"   ⚠️  These files are PERMANENTLY gone and cannot be recovered")

        return total_images, deleted_count

    def verify_dataset(self, split: str = "train") -> dict:
        """
        Verify dataset integrity and return statistics

        PURPOSE:
            Check if your dataset is "clean" (all images have matching labels)
            Returns detailed statistics about the dataset state

        WHY THIS IS USEFUL:
            - See how many images lack labels before cleaning
            - Verify dataset is clean after cleaning
            - Track dataset statistics

        Args:
            split: Dataset split to verify ('train' or 'test')

        Returns:
            Dictionary containing:
                - split: which split was checked ('train' or 'test')
                - num_images: total number of image files in the folder
                - num_labels: total number of label (.txt) files in the folder
                - unpaired_images: how many images don't have labels
                - is_clean: True if all images have labels, False otherwise
                - match_percentage: what percentage of images have labels

        Example Output:
            {
                'split': 'train',
                'num_images': 83791,      # 83,791 total images
                'num_labels': 82085,      # 82,085 total labels
                'unpaired_images': 1706,  # 1,706 images without labels
                'is_clean': False,        # Dataset needs cleaning
                'match_percentage': 97.96 # 97.96% have labels
            }
        """
        images_path = self.images_dir / split
        labels_path = self.labels_dir / split

        # ========================================
        # COUNT IMAGES
        # ========================================
        # Count all image files in the images folder
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        num_images = sum(len(list(images_path.glob(f"*{ext}")))
                        for ext in image_extensions)

        # ========================================
        # COUNT LABELS
        # ========================================
        # Count all .txt files in the labels folder
        num_labels = len(list(labels_path.glob("*.txt")))

        # ========================================
        # FIND UNPAIRED IMAGES
        # ========================================
        # Get list of images without matching labels
        unpaired = self.find_unpaired_images(split)

        # ========================================
        # BUILD STATISTICS DICTIONARY
        # ========================================
        # Calculate match percentage: what % of images have labels
        match_percentage = (num_labels / num_images * 100) if num_images > 0 else 0

        stats = {
            'split': split,
            'num_images': num_images,
            'num_labels': num_labels,
            'unpaired_images': len(unpaired),
            'is_clean': len(unpaired) == 0,  # Clean = all images have labels
            'match_percentage': match_percentage
        }

        return stats

    def print_report(self, stats: dict):
        """
        Print a nicely formatted report of dataset statistics

        PURPOSE:
            Display dataset statistics in an easy-to-read format
            Shows total images, labels, unpaired images, and match percentage

        Args:
            stats: Dictionary from verify_dataset() function

        Output Example:
            ============================================================
            📊 DATASET REPORT - TRAIN SPLIT
            ============================================================
              Total Images:      83,791
              Total Labels:      82,085
              Unpaired Images:   1,706  ← These will be DELETED
              Match Percentage:  97.96%
              Status:            ⚠️  NEEDS CLEANING
            ============================================================
        """
        print("\n" + "="*60)
        print(f"📊 DATASET REPORT - {stats['split'].upper()} SPLIT")
        print("="*60)
        print(f"  Total Images:      {stats['num_images']:,}")
        print(f"  Total Labels:      {stats['num_labels']:,}")
        print(f"  Unpaired Images:   {stats['unpaired_images']:,}")
        print(f"  Match Percentage:  {stats['match_percentage']:.2f}%")
        print(f"  Status:            {'✅ CLEAN' if stats['is_clean'] else '⚠️  NEEDS CLEANING'}")
        print("="*60)


def main():
    """
    Main execution function - orchestrates the entire cleaning process

    WORKFLOW:
        1. Analyze current dataset state (show before statistics)
        2. Run DRY RUN to preview what will be deleted (SAFETY CHECK - no changes made)
        3. Ask user for confirmation with CLEAR WARNING about permanent deletion
        4. If user confirms with "yes":
            a. PERMANENTLY DELETE unpaired images from train split
            b. PERMANENTLY DELETE unpaired images from test split
            c. Verify cleaned dataset (show after statistics)
            d. Show final summary
        5. If user types anything else: exit without making changes

    ⚠️  SAFETY FEATURES:
        - Always shows DRY RUN preview before deleting anything
        - Requires explicit user confirmation ("yes")
        - Shows clear warnings about permanent deletion
        - User can cancel by typing "no" or anything other than "yes"

    ⚠️  WARNING:
        - Files are PERMANENTLY DELETED (not moved to backup)
        - Deleted files CANNOT be recovered
        - Make sure you have backups if you're unsure
    """

    # ========================================
    # CONFIGURATION
    # ========================================
    # Path to your dataset root folder
    # ⚠️  IMPORTANT: Verify this path matches your dataset location!
    dataset_root = "dataset/content/UA-DETRAC/DETRAC_Upload"

    print("🧹 DATASET CLEANING TOOL FOR YOLO")
    print("="*60)
    print("⚠️  WARNING: This tool PERMANENTLY DELETES unpaired images")
    print("="*60)

    # Create cleaner instance
    cleaner = DatasetCleaner(dataset_root)

    # ========================================
    # STEP 1: ANALYZE CURRENT STATE (BEFORE CLEANING)
    # ========================================
    print("\n📋 STEP 1: Analyzing current dataset state...")
    print("   (This will show how many images lack labels)")

    # Check train split - how many images, labels, and unpaired images
    train_stats_before = cleaner.verify_dataset("train")

    # Check test split - how many images, labels, and unpaired images
    test_stats_before = cleaner.verify_dataset("test")

    # Print detailed reports for both splits
    cleaner.print_report(train_stats_before)
    cleaner.print_report(test_stats_before)

    # ========================================
    # STEP 2: DRY RUN (PREVIEW ONLY - NO DELETION)
    # ========================================
    print("\n📋 STEP 2: Running DRY RUN (preview mode - no files will be deleted)...")
    print("   (This shows you EXACTLY what will be deleted)")

    # dry_run=True means: just show what would be deleted, don't actually delete
    # This is a CRITICAL SAFETY FEATURE - always preview before deletion!
    cleaner.clean_split("train", dry_run=True)
    cleaner.clean_split("test", dry_run=True)

    # ========================================
    # STEP 3: ASK USER FOR CONFIRMATION
    # ========================================
    # Show clear warning and get user's decision
    print("\n" + "="*60)
    print("⚠️  ⚠️  ⚠️  WARNING - PERMANENT DELETION ⚠️  ⚠️  ⚠️")
    print("="*60)
    print("You are about to PERMANENTLY DELETE all unpaired images!")
    print("")
    print("What will happen:")
    print(f"  • Train: {train_stats_before['unpaired_images']:,} images will be DELETED")
    print(f"  • Test:  {test_stats_before['unpaired_images']:,} images will be DELETED")
    print(f"  • Total: {train_stats_before['unpaired_images'] + test_stats_before['unpaired_images']:,} images will be DELETED")
    print("")
    print("⚠️  THESE FILES CANNOT BE RECOVERED AFTER DELETION!")
    print("⚠️  Make sure you have reviewed the file list above!")
    print("="*60)

    # Get user's decision
    # Only "yes" or "y" will proceed - anything else cancels
    response = input("\nType 'yes' to PERMANENTLY DELETE these files (or 'no' to cancel): ").strip().lower()

    # ========================================
    # STEP 4: EXECUTE DELETION (if user confirmed)
    # ========================================
    if response in ['yes', 'y']:
        print("\n📋 STEP 3: Deleting unpaired images...")
        print("   ⚠️  Deletion in progress - this cannot be undone!")

        # Clean train split
        # dry_run=False: actually perform the PERMANENT DELETION
        train_total, train_deleted = cleaner.clean_split("train", dry_run=False)

        # Clean test split
        # dry_run=False: actually perform the PERMANENT DELETION
        test_total, test_deleted = cleaner.clean_split("test", dry_run=False)

        # ========================================
        # STEP 5: VERIFY RESULTS (AFTER CLEANING)
        # ========================================
        print("\n📋 STEP 4: Verifying cleaned dataset...")
        print("   (Checking that all remaining images have labels)")

        # Check dataset state after cleaning
        train_stats_after = cleaner.verify_dataset("train")
        test_stats_after = cleaner.verify_dataset("test")

        # Print final statistics - should show is_clean=True now
        cleaner.print_report(train_stats_after)
        cleaner.print_report(test_stats_after)

        # ========================================
        # STEP 6: SHOW SUMMARY
        # ========================================
        print("\n" + "="*60)
        print("🎉 CLEANING COMPLETE!")
        print("="*60)
        print(f"  Train: {train_deleted:,} images PERMANENTLY DELETED")
        print(f"  Test:  {test_deleted:,} images PERMANENTLY DELETED")
        print(f"  Total: {train_deleted + test_deleted:,} images PERMANENTLY DELETED")
        print("="*60)
        print(f"\n  Remaining Train Images: {train_stats_after['num_images']:,}")
        print(f"  Remaining Test Images:  {test_stats_after['num_images']:,}")
        print("="*60)
        print("\n✅ Dataset is now clean and ready for YOLOv8 training!")
        print("   All remaining images have matching label files")

    else:
        # User cancelled - no files were deleted
        print("\n❌ Cleaning cancelled by user.")
        print("   No files were deleted - your dataset is unchanged")
        print("   You can run this script again anytime")


# ========================================
# SCRIPT ENTRY POINT
# ========================================
if __name__ == "__main__":
    # This block runs when you execute: python clean_dataset.py
    # It calls the main() function which orchestrates the entire process
    main()
