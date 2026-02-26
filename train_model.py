import pandas as pd
import numpy as np
import os
import joblib
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================
# LOAD DATASET
# ==========================
data = pd.read_csv("Crop_recommendation - Copy.csv")

X = data.drop("label", axis=1)
y = data["label"]

# ==========================
# STRATIFIED SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# HYPERPARAMETER TUNING
# ==========================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=3,
    n_jobs=-1
)

print("Training model with GridSearch...")

start_time = time.time()
grid.fit(X_train, y_train)
end_time = time.time()

training_time = round(end_time - start_time, 3)

rf = grid.best_estimator_

print("Best Parameters:", grid.best_params_)

# ==========================
# CROSS VALIDATION
# ==========================
cv_scores = cross_val_score(rf, X, y, cv=5)

print("Cross Validation Scores:", cv_scores)
print("Average CV Accuracy:", round(cv_scores.mean(), 4))

# ==========================
# PREDICTIONS
# ==========================
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

# ==========================
# OVERFITTING CHECK
# ==========================
train_acc = rf.score(X_train, y_train)
test_acc = rf.score(X_test, y_test)

print("\n===== MODEL PERFORMANCE =====")
print("Training Accuracy :", round(train_acc,4))
print("Testing Accuracy  :", round(test_acc,4))
print("Accuracy          :", round(accuracy,4))
print("Precision         :", round(precision,4))
print("Recall (Sensitivity):", round(recall,4))
print("F1 Score          :", round(f1,4))
print("Training Time (sec):", training_time)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ==========================
# CONFUSION MATRIX
# ==========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=False, cmap='Greens')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ==========================
# FEATURE IMPORTANCE
# ==========================
importance = rf.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ==========================
# SAVE MODEL
# ==========================
if not os.path.exists("models"):
    os.makedirs("models")

joblib.dump(rf, "models/crop_model.pkl")

print("\nModel Saved Successfully!")
print("Classes in model:", rf.classes_)
print("Model trained on:", datetime.datetime.now())

# ==========================
# SAVE METRICS TO FILE
# ==========================
with open("model_metrics.txt", "w") as f:
    f.write(f"Training Accuracy: {train_acc}\n")
    f.write(f"Testing Accuracy: {test_acc}\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Training Time: {training_time}\n")

print("Metrics saved to model_metrics.txt")