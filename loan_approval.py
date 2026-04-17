# ==========================================
# LOAN APPROVAL PREDICTION
# Logistic Regression vs SVM
# ROC + Metrics Bar Chart
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_curve,
    auc
)

# =========================
# 1. Load Dataset
# =========================
data = pd.read_csv("loan_approval_dataset.csv")
data.columns = data.columns.str.strip()

# Drop loan_id if exists
if "loan_id" in data.columns:
    data = data.drop("loan_id", axis=1)

# Clean categorical spaces
for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].str.strip()

# Encode target
data["loan_status"] = data["loan_status"].map({
    "Approved": 1,
    "Rejected": 0
})

# Fill missing values
data = data.ffill()

# One-hot encoding
data = pd.get_dummies(data, drop_first=True)

# =========================
# 2. Train-Test Split
# =========================
X = data.drop("loan_status", axis=1)
y = data["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 3. Logistic Regression
# =========================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_log).ravel()

accuracy_log = accuracy_score(y_test, y_pred_log)
sensitivity_log = recall_score(y_test, y_pred_log)
specificity_log = tn / (tn + fp)
f1_log = f1_score(y_test, y_pred_log)

fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
auc_log = auc(fpr_log, tpr_log)

# =========================
# 4. SVM
# =========================
svm_model = SVC(kernel="rbf", probability=True)
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_svm).ravel()

accuracy_svm = accuracy_score(y_test, y_pred_svm)
sensitivity_svm = recall_score(y_test, y_pred_svm)
specificity_svm = tn / (tn + fp)
f1_svm = f1_score(y_test, y_pred_svm)

fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
auc_svm = auc(fpr_svm, tpr_svm)
# =========================
# 5. Print Results
# =========================
print("===== Logistic Regression =====")
print("Accuracy:", accuracy_log)
print("Sensitivity:", sensitivity_log)
print("Specificity:", specificity_log)
print("F1 Score:", f1_log)
print("AUC:", auc_log)

print("\n===== SVM =====")
print("Accuracy:", accuracy_svm)
print("Sensitivity:", sensitivity_svm)
print("Specificity:", specificity_svm)
print("F1 Score:", f1_svm)
print("AUC:", auc_svm)

# ==========================================
# 6. ROC Curve (Separate Image)
# ==========================================
plt.figure(figsize=(6,5))
plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC={auc_log:.3f})")
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC={auc_svm:.3f})")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)

plt.savefig("ROC_Curve_Comparison.png", dpi=300)
plt.show()


# ==========================================
# 7. Metrics Comparison Bar Chart
# (Accuracy, Sensitivity, Specificity, F1, AUC)
# ==========================================

metrics = ["Accuracy", "Sensitivity", "Specificity", "F1-Score", "AUC"]

log_values = [accuracy_log, sensitivity_log, specificity_log, f1_log, auc_log]
svm_values = [accuracy_svm, sensitivity_svm, specificity_svm, f1_svm, auc_svm]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, log_values, width, label="Logistic Regression")
plt.bar(x + width/2, svm_values, width, label="SVM")

plt.xticks(x, metrics)
plt.ylabel("Score")
plt.title("Performance Metrics Comparison")
plt.ylim(0, 1)
plt.legend()

plt.savefig("Performance_Metrics_Comparison.png", dpi=300)
plt.show()

joblib.dump(svm_model, "loan_model.pkl")      # or log_model
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns, "model_columns.pkl")