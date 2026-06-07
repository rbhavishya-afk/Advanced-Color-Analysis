# =====================================================================
# HANDWRITTEN DIGIT RECOGNITION SYSTEM (SINGLE-FILE SCRIPT)
# =====================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, f1_score

# --- STEP 1: LOAD AND PREPROCESS DATA ---
print("\n=== Step 1: Fetching and Standardizing MNIST Dataset ===")
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values from [0, 255] to [0.0, 1.0]
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshape datasets to match CNN 4D input dimensions (Batch, Height, Width, Channel)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print(f"Dataset Loaded Successfully! Training size: {X_train.shape}, Test size: {X_test.shape}")


# --- STEP 2: DESIGN THE ARCHITECTURE (Model Design Rubric) ---
print("\n=== Step 2: Compiling Convolutional Neural Network (CNN) ===")
model = models.Sequential([
    # First Feature Extraction Layer
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Second Feature Extraction Layer
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Classification Head
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),  # Regularization to prevent overfitting
    layers.Dense(10, activation='softmax')  # 10 output classes for digits 0-9
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# --- STEP 3: TRAIN THE NETWORK (Performance Rubric) ---
print("\n=== Step 3: Model Optimization and Training ===")
# Train for 5 epochs using a batch size of 64
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)


# --- STEP 4: SAVE EXPORTABLE MODEL ARTIFACTS (Deliverables Rubric) ---
print("\n=== Step 4: Exporting Weights to Disk ===")
output_dir = "saved_models"
os.makedirs(output_dir, exist_ok=True)
model_file_path = os.path.join(output_dir, "digit_recog_cnn.h5")
model.save(model_file_path)
print(f"[ARTIFACT SUCCESS] Saved weights file located at: {model_file_path}")


# --- STEP 5: PERFORMANCE EVALUATION (Reporting Rubric) ---
print("\n=== Step 5: Computing Testing Metrics ===")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

# Generate Class Predictions
predictions = model.predict(X_test, verbose=0)
y_pred = np.argmax(predictions, axis=1)

print("\n" + "="*60)
print("             DIGIT CLASSIFICATION METRICS REPORT")
print("="*60)
print(classification_report(y_test, y_pred))
print("="*60)
print(f"Overall Network Accuracy: {test_acc*100:.2f}%")
print(f"Macro Average F1-Score: {f1_score(y_test, y_pred, average='macro'):.4f}\n")


# --- STEP 6: VISUALIZATION MATRIX (Sample Predictions) ---
print("=== Step 6: Rendering Evaluation Plots ===")
# Plot 1: Accuracy and Loss Charts
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Training Accuracy Trend')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Training Loss Trend')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: Grid of Test Sample Predictions
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    color = 'green' if y_pred[i] == y_test[i] else 'red'
    plt.title(f"Pred: {y_pred[i]}\nTrue: {y_test[i]}", color=color)
    plt.axis('off')
plt.suptitle("Sample Test Prediction Interface\n(Green = Match | Red = Error)", fontsize=14)
plt.tight_layout()
plt.show()