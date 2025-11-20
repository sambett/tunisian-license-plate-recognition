"""
Tunisian License Plate Recognition - CRNN Training Script V2
Improved version with data augmentation and regularization to prevent overfitting
"""

import os
import csv
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_DIR = "license_plates_recognition_train"
CSV_PATH = "license_plates_recognition_train.csv"
MODEL_SAVE_PATH = "tunisian_plate_crnn_model_v2.h5"
CHECKPOINT_PATH = "best_model_checkpoint_v2.h5"

CHARACTERS = "0123456789TN "
NUM_CLASSES = len(CHARACTERS) + 1

# Image dimensions
IMG_WIDTH = 128
IMG_HEIGHT = 64

# Training hyperparameters - IMPROVED
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15

# Regularization - NEW
DROPOUT_RATE = 0.5  # Aggressive dropout
L2_REG = 0.001  # Weight decay

# CTC parameters
MAX_LABEL_LENGTH = 12

# Data augmentation - NEW
AUGMENT_PROB = 0.7  # Probability of augmentation

print(f"Character set: '{CHARACTERS}'")
print(f"Number of classes (including CTC blank): {NUM_CLASSES}")
print(f"Using aggressive regularization to prevent overfitting")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def char_to_num(char):
    if char in CHARACTERS:
        return CHARACTERS.index(char)
    return -1

def num_to_char(num):
    if 0 <= num < len(CHARACTERS):
        return CHARACTERS[num]
    return ""

def encode_label(text):
    return [char_to_num(c) for c in text if char_to_num(c) != -1]

def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    output_text = []
    for res in results:
        res = res.numpy()
        text = "".join([num_to_char(int(idx)) for idx in res if idx != -1])
        output_text.append(text)
    return output_text

# ============================================================================
# DATA AUGMENTATION - KEY FOR PREVENTING OVERFITTING
# ============================================================================

def augment_image(img):
    """Apply random augmentation to prevent overfitting"""

    if np.random.rand() > AUGMENT_PROB:
        return img

    # Random brightness (wider range)
    if np.random.rand() > 0.3:
        factor = np.random.uniform(0.5, 1.5)
        img = np.clip(img * factor, 0, 1)

    # Random contrast
    if np.random.rand() > 0.3:
        factor = np.random.uniform(0.7, 1.3)
        mean = np.mean(img)
        img = np.clip((img - mean) * factor + mean, 0, 1)

    # Random noise
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 0.05, img.shape)
        img = np.clip(img + noise, 0, 1)

    # Random blur
    if np.random.rand() > 0.7:
        kernel_size = np.random.choice([3, 5])
        img_uint8 = (img * 255).astype(np.uint8)
        img_uint8 = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), 0)
        img = img_uint8.astype(np.float32) / 255.0

    # Random erosion/dilation (morphological)
    if np.random.rand() > 0.8:
        kernel = np.ones((2, 2), np.uint8)
        img_uint8 = (img * 255).astype(np.uint8)
        if np.random.rand() > 0.5:
            img_uint8 = cv2.erode(img_uint8, kernel, iterations=1)
        else:
            img_uint8 = cv2.dilate(img_uint8, kernel, iterations=1)
        img = img_uint8.astype(np.float32) / 255.0

    # Random rotation (small angles)
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-8, 8)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Random scale/zoom
    if np.random.rand() > 0.7:
        scale = np.random.uniform(0.9, 1.1)
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (new_w, new_h))

        # Crop or pad back to original size
        if scale > 1:
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            img = img_resized[start_h:start_h+h, start_w:start_w+w]
        else:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            img = np.pad(img_resized, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)),
                        mode='edge')

    # Ensure correct shape
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    return img.astype(np.float32)

def preprocess_image(img_path, augment=False):
    """Load and preprocess image with optional augmentation"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0

    if augment:
        img = augment_image(img)

    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    return img

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(csv_path, dataset_dir):
    image_paths = []
    labels = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row['img_id']
            text = row['text']
            img_path = os.path.join(dataset_dir, img_name)

            if os.path.exists(img_path):
                image_paths.append(img_path)
                labels.append(text)
            else:
                print(f"Warning: Image not found: {img_path}")

    print(f"Loaded {len(image_paths)} images from CSV")
    return image_paths, labels

# ============================================================================
# DATASET PIPELINE
# ============================================================================

def create_dataset(image_paths, labels, batch_size, shuffle=True, augment=False):
    """Create TensorFlow dataset with optional augmentation"""

    def generator():
        indices = np.arange(len(image_paths))
        if shuffle:
            np.random.shuffle(indices)

        for idx in indices:
            img = preprocess_image(image_paths[idx], augment=augment)

            label = encode_label(labels[idx])
            label = np.array(label, dtype=np.int32)

            label_length = len(label)
            label_padded = np.pad(label, (0, MAX_LABEL_LENGTH - len(label)),
                                 constant_values=-1)

            yield {
                'image': img,
                'label': label_padded,
                'label_length': np.array([label_length], dtype=np.int32)
            }, np.zeros(1)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            {
                'image': tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
                'label': tf.TensorSpec(shape=(MAX_LABEL_LENGTH,), dtype=tf.int32),
                'label_length': tf.TensorSpec(shape=(1,), dtype=tf.int32)
            },
            tf.TensorSpec(shape=(1,), dtype=tf.float32)
        )
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# ============================================================================
# MODEL ARCHITECTURE - SIMPLIFIED WITH REGULARIZATION
# ============================================================================

def build_crnn_model(img_width, img_height, num_classes):
    """Build CRNN with stronger regularization"""

    input_img = layers.Input(shape=(img_height, img_width, 1), name="image")

    # CNN with L2 regularization and more dropout
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                     kernel_regularizer=regularizers.l2(L2_REG))(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)  # Add dropout after pooling

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                     kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same",
                     kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same",
                     kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Dropout(0.3)(x)

    # Reshape for RNN
    new_shape = ((img_width // 4), (img_height // 16) * 128)
    x = layers.Reshape(target_shape=new_shape)(x)

    # Dense with dropout
    x = layers.Dense(64, activation="relu",
                    kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    # Bidirectional LSTMs with recurrent dropout
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True,
                                         dropout=0.3, recurrent_dropout=0.1))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True,
                                         dropout=0.3, recurrent_dropout=0.1))(x)

    # Output
    x = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=input_img, outputs=x, name="CRNN")
    return model

# ============================================================================
# CTC LOSS
# ============================================================================

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

def build_training_model(img_width, img_height, num_classes):
    input_img = layers.Input(shape=(img_height, img_width, 1), name="image")
    labels = layers.Input(shape=(MAX_LABEL_LENGTH,), dtype="int32", name="label")
    label_length = layers.Input(shape=(1,), dtype="int32", name="label_length")

    crnn = build_crnn_model(img_width, img_height, num_classes)
    y_pred = crnn(input_img)

    pred_length_value = img_width // 4

    def compute_pred_length(y_pred):
        batch_size = tf.shape(y_pred)[0]
        return tf.ones((batch_size, 1), dtype="int32") * pred_length_value

    pred_length = layers.Lambda(compute_pred_length, name="pred_length")(y_pred)

    ctc_layer = CTCLayer(name="ctc_loss")
    output = ctc_layer(labels, y_pred, pred_length, label_length)

    model = keras.Model(
        inputs=[input_img, labels, label_length],
        outputs=output,
        name="CRNN_CTC"
    )

    return model, crnn

# ============================================================================
# METRICS CALLBACK
# ============================================================================

class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, validation_data, crnn_model):
        super().__init__()
        self.validation_images, self.validation_labels = validation_data
        self.crnn_model = crnn_model
        self.best_plate_acc = 0

    def on_epoch_end(self, epoch, logs=None):
        # Evaluate on validation set
        sample_size = min(100, len(self.validation_images))
        indices = np.random.choice(len(self.validation_images), sample_size, replace=False)

        correct_chars = 0
        total_chars = 0
        correct_plates = 0

        for idx in indices:
            img = preprocess_image(self.validation_images[idx], augment=False)
            img_batch = np.expand_dims(img, axis=0)

            pred = self.crnn_model.predict(img_batch, verbose=0)
            pred_text = decode_prediction(pred)[0]
            true_text = self.validation_labels[idx]

            for p, t in zip(pred_text, true_text):
                if p == t:
                    correct_chars += 1
                total_chars += 1

            if pred_text == true_text:
                correct_plates += 1

        char_acc = correct_chars / total_chars if total_chars > 0 else 0
        plate_acc = correct_plates / sample_size

        # Track best
        if plate_acc > self.best_plate_acc:
            self.best_plate_acc = plate_acc

        print(f"\n[Val] Char Acc: {char_acc:.4f} | Plate Acc: {plate_acc:.4f} | Best: {self.best_plate_acc:.4f}")

        logs['val_char_accuracy'] = char_acc
        logs['val_plate_accuracy'] = plate_acc

# ============================================================================
# TRAINING
# ============================================================================

def train_model():
    print("\n" + "="*70)
    print("CRNN TRAINING V2 - WITH DATA AUGMENTATION & REGULARIZATION")
    print("="*70 + "\n")

    # Load data
    print("Loading data...")
    image_paths, labels = load_data(CSV_PATH, DATASET_DIR)
    print(f"Total samples: {len(image_paths)}")

    # Split
    train_images, val_images, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=VALIDATION_SPLIT, random_state=42
    )

    print(f"Train samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}\n")

    # Build model
    print("Building CRNN model with regularization...")
    training_model, crnn_model = build_training_model(IMG_WIDTH, IMG_HEIGHT, NUM_CLASSES)
    crnn_model.summary()

    # Compile with gradient clipping
    optimizer = keras.optimizers.Adam(LEARNING_RATE, clipnorm=1.0)
    training_model.compile(optimizer=optimizer)

    # Create datasets - AUGMENTATION ON TRAINING ONLY
    print("\nCreating data pipelines with augmentation...")
    train_dataset = create_dataset(train_images, train_labels, BATCH_SIZE,
                                   shuffle=True, augment=True)  # Augment training
    val_dataset = create_dataset(val_images, val_labels, BATCH_SIZE,
                                shuffle=False, augment=False)  # No augment for val

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            CHECKPOINT_PATH,
            monitor='val_plate_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_plate_accuracy',
            mode='max',
            patience=20,  # More patience
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        MetricsCallback((val_images, val_labels), crnn_model)
    ]

    # Train
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")

    history = training_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    print(f"\nSaving final model to {MODEL_SAVE_PATH}")
    crnn_model.save(MODEL_SAVE_PATH)

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)

    return history, crnn_model

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')

    history, model = train_model()

    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print(f"Best checkpoint: {CHECKPOINT_PATH}")
