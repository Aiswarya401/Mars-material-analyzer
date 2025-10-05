import tensorflow as tf
from tensorflow.keras import layers, models
import json
import sys
import os

# --- Configuration ---
# *** FIX APPLIED HERE: Changed 'data/train' to 'data/Train' based on file structure. ***
train_dir = "data/Train" 
val_dir = "data/val"
img_size = (224, 224)
batch_size = 16

# --- Dataset Loading ---
print(f"Attempting to load datasets from {train_dir} and {val_dir}...")

try:
    # Load datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size
    )
except Exception as e:
    # Provide helpful error message if data loading fails
    print(f"\n--- ❌ ERROR: Dataset Loading Failed ---", file=sys.stderr)
    print(f"1. Check directory names (case-sensitivity is important: '{train_dir}' and '{val_dir}').", file=sys.stderr)
    print(f"2. Ensure these directories contain subfolders (e.g., 'Aluminium') with images.", file=sys.stderr)
    print(f"Original error: {e}", file=sys.stderr)
    sys.exit(1)


# Get class names
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Classes found: {class_names} (Total: {num_classes})")

if num_classes == 0:
    print("\n--- ❌ ERROR: No Classes Found ---", file=sys.stderr)
    print("Please ensure your data directories contain subdirectories (one for each class) with images.", file=sys.stderr)
    sys.exit(1)


# --- Preprocessing and Optimization ---
# Normalize pixel values
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Prefetch for performance
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Save class names to JSON
with open("materials.json", "w") as f:
    json.dump(class_names, f)
print("✅ Class names saved to materials.json.")

# --- Model Definition and Training ---
print("\n--- Defining Model Architecture ---")
try:
    # Load base model (MobileNetV2)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_size + (3,),
        include_top=False,
        weights="imagenet"
    )
except Exception as e:
    print(f"\n--- ❌ ERROR: Failed to load MobileNetV2 ---", file=sys.stderr)
    print(f"Error details: {e}", file=sys.stderr)
    sys.exit(1)

base_model.trainable = False

# Add classifier head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train model (Phase 1: Feature Extraction)
print("\n--- Starting Initial Training (Phase 1/2) ---")
model.fit(train_ds, validation_data=val_ds, epochs=5)

# Optional fine-tuning (Phase 2)
print("\n--- Starting Fine-Tuning (Phase 2/2) ---")
base_model.trainable = True
fine_tune_at = int(len(base_model.layers) * 0.7)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=5)

# Save trained model
model.save("material_model.h5")
print("\n✅ Model saved as material_model.h5")
