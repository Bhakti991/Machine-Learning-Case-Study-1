# Handwritten Digit Recognition Case Study
# Dataset: MNIST (28x28 grayscale digits)
# Models: Logistic Regression, SVM, KNN, Random Forest, MLP (sklearn), Simple CNN (Keras)
# Evaluation: accuracy, classification report, confusion matrix, comparison plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# For the CNN
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ------------------------
# 1. Load the dataset
# ------------------------
# Option A: use Keras MNIST (smaller / ready)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Stack for convenience (we'll split again later for classical models)
X = np.vstack([x_train, x_test])
y = np.hstack([y_train, y_test])

print("Total samples:", X.shape, "Labels:", np.unique(y))

# Visualize first 6 digits
def show_samples(X, y, n=6):
    plt.figure(figsize=(10,2))
    for i in range(n):
        plt.subplot(1,n,i+1)
        plt.imshow(X[i], cmap='gray')
        plt.title(str(y[i]))
        plt.axis('off')
    plt.show()

show_samples(X, y)

# ------------------------
# 2. Prepare data for classical ML
# ------------------------
# Flatten images to vectors
n_samples = X.shape[0]
X_flat = X.reshape((n_samples, -1))  # shape (70000, 784)

# Optionally reduce dimension with PCA for speed (uncomment if you want)
# pca = PCA(n_components=0.95, svd_solver='full')  # keep 95% variance
# X_reduced = pca.fit_transform(X_flat)
# print("Reduced shape:", X_reduced.shape)
# For this run we'll use flattened data (or you can use X_reduced)

# Train/test split for classical ML (stratify to keep label proportions)
X_train, X_val, y_train_cls, y_val_cls = train_test_split(
    X_flat, y, test_size=0.15, random_state=42, stratify=y)

print("Classical ML split:", X_train.shape, X_val.shape)

# Standardize features (important for many algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
X_val_scaled = scaler.transform(X_val.astype(np.float64))

# ------------------------
# 3. Train classical ML models
# ------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, solver='saga', multi_class='multinomial', n_jobs=-1),
    "SVM": SVC(kernel='rbf', gamma='scale'),  # change C/gamma for tuning
    "KNN": KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
    "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(128,), max_iter=30, verbose=False, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name} ...")
    t0 = time.time()
    # Use scaled data for all models (SVM, LR, MLP benefit; RF and KNN can work without but scaled is fine)
    model.fit(X_train_scaled, y_train_cls)
    preds = model.predict(X_val_scaled)
    acc = accuracy_score(y_val_cls, preds)
    elapsed = time.time() - t0
    print(f"{name} done — Accuracy: {acc:.4f}, Time: {elapsed:.1f}s")
    results[name] = {
        "model": model,
        "accuracy": acc,
        "preds": preds,
        "time": elapsed
    }

# Print classification report for the best classical model
best_cl = max(results.items(), key=lambda kv: kv[1]['accuracy'])[0]
print(f"\nBest classical model: {best_cl} with accuracy {results[best_cl]['accuracy']:.4f}")
print(classification_report(y_val_cls, results[best_cl]['preds']))

# Confusion matrix plot for best classical model
cm = confusion_matrix(y_val_cls, results[best_cl]['preds'])
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix — {best_cl}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ------------------------
# 4. Prepare data for CNN
# ------------------------
# Use original images (28x28). Normalize and one-hot encode labels.
X = X.astype('float32') / 255.0
y_cat = to_categorical(y, num_classes=10)

# Split for deep model training/validation
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    X, y_cat, test_size=0.15, random_state=42, stratify=y)

# expand dims for channels (batch, 28, 28, 1)
X_train_cnn = np.expand_dims(X_train_cnn, -1)
X_test_cnn = np.expand_dims(X_test_cnn, -1)
print("CNN train shape:", X_train_cnn.shape, "CNN test shape:", X_test_cnn.shape)

# ------------------------
# 5. Build and train a simple CNN
# ------------------------
def build_simple_cnn(input_shape=(28,28,1), num_classes=10):
    model = Sequential([
        Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

cnn = build_simple_cnn()
cnn.summary()

# Train (use small epochs for demo; increase for better performance)
epochs = 6
batch_size = 128

history = cnn.fit(
    X_train_cnn, y_train_cnn,
    validation_data=(X_test_cnn, y_test_cnn),
    epochs=epochs,
    batch_size=batch_size,
    verbose=2
)

# Evaluate CNN
cnn_eval = cnn.evaluate(X_test_cnn, y_test_cnn, verbose=0)
print(f"\nCNN Test Loss: {cnn_eval[0]:.4f}  Test Accuracy: {cnn_eval[1]:.4f}")

# ------------------------
# 6. Compare results
# ------------------------
# Collect results
model_names = []
accuracies = []
times = []
for name, info in results.items():
    model_names.append(name)
    accuracies.append(info['accuracy'])
    times.append(info['time'])

model_names.append('CNN')
accuracies.append(cnn_eval[1])
times.append(np.nan)  # training time for CNN is available via history but keeping simple

# Comparison bar chart
plt.figure(figsize=(10,5))
sns.barplot(x=accuracies, y=model_names)
plt.xlim(0,1)
plt.xlabel('Accuracy')
plt.title('Model Comparison (Validation/Test Accuracy)')
for i, v in enumerate(accuracies):
    plt.text(v + 0.01, i, f"{v:.4f}")
plt.show()

# ------------------------
# 7. Show some CNN prediction samples
# ------------------------
# pick some examples
n_show = 12
test_preds = cnn.predict(X_test_cnn[:n_show])
test_preds_labels = np.argmax(test_preds, axis=1)
actual_labels = np.argmax(y_test_cnn[:n_show], axis=1)

plt.figure(figsize=(12,3))
for i in range(n_show):
    plt.subplot(2, 6, i+1)
    plt.imshow(X_test_cnn[i].reshape(28,28), cmap='gray')
    plt.title(f"GT:{actual_labels[i]} Pred:{test_preds_labels[i]}")
    plt.axis('off')
plt.show()

# ------------------------
# 8. Save models (optional)
# ------------------------
# from joblib import dump
# dump(results['RandomForest']['model'], 'rf_mnist.joblib')
# cnn.save('cnn_mnist.h5')

# ------------------------
# 9. Notes & next steps
# ------------------------
# - For classical models: try PCA (fast), hyperparameter tuning (GridSearchCV), and balancing.
# - For CNN: increase epochs, add data augmentation, try deeper models like LeNet/ResNet.
# - For real deployment: persist scaler + model, and serve with an API.
