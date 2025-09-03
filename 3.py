import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

df = pd.read_csv("data_metrics.csv")

thresh = 0.5
df["predicted_RF"] = (df.model_RF >= thresh).astype(int)
df["predicted_LR"] = (df.model_LR >= thresh).astype(int)


def liashenko_find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))

def liashenko_find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))

def liashenko_find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))

def liashenko_find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))

def liashenko_find_conf_matrix_values(y_true, y_pred):
    TP = liashenko_find_TP(y_true, y_pred)
    FN = liashenko_find_FN(y_true, y_pred)
    FP = liashenko_find_FP(y_true, y_pred)
    TN = liashenko_find_TN(y_true, y_pred)
    return TP, FN, FP, TN

def liashenko_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = liashenko_find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

def liashenko_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = liashenko_find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

def liashenko_recall_score(y_true, y_pred):
    TP, FN, FP, TN = liashenko_find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def liashenko_precision_score(y_true, y_pred):
    TP, FN, FP, TN = liashenko_find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def liashenko_f1_score(y_true, y_pred):
    recall = liashenko_recall_score(y_true, y_pred)
    precision = liashenko_precision_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# ------------------- перевірки -------------------

# Confusion matrix
assert np.array_equal(
    liashenko_confusion_matrix(df.actual_label.values, df.predicted_RF.values),
    confusion_matrix(df.actual_label.values, df.predicted_RF.values)
), "confusion_matrix не співпадає для RF"

assert np.array_equal(
    liashenko_confusion_matrix(df.actual_label.values, df.predicted_LR.values),
    confusion_matrix(df.actual_label.values, df.predicted_LR.values)
), "confusion_matrix не співпадає для LR"

# Accuracy
assert np.isclose(
    liashenko_accuracy_score(df.actual_label.values, df.predicted_RF.values),
    accuracy_score(df.actual_label.values, df.predicted_RF.values)
)
assert np.isclose(
    liashenko_accuracy_score(df.actual_label.values, df.predicted_LR.values),
    accuracy_score(df.actual_label.values, df.predicted_LR.values)
)

# Recall
assert np.isclose(
    liashenko_recall_score(df.actual_label.values, df.predicted_RF.values),
    recall_score(df.actual_label.values, df.predicted_RF.values)
)
assert np.isclose(
    liashenko_recall_score(df.actual_label.values, df.predicted_LR.values),
    recall_score(df.actual_label.values, df.predicted_LR.values)
)

# Precision
assert np.isclose(
    liashenko_precision_score(df.actual_label.values, df.predicted_RF.values),
    precision_score(df.actual_label.values, df.predicted_RF.values)
)
assert np.isclose(
    liashenko_precision_score(df.actual_label.values, df.predicted_LR.values),
    precision_score(df.actual_label.values, df.predicted_LR.values)
)

# F1
assert np.isclose(
    liashenko_f1_score(df.actual_label.values, df.predicted_RF.values),
    f1_score(df.actual_label.values, df.predicted_RF.values)
)
assert np.isclose(
    liashenko_f1_score(df.actual_label.values, df.predicted_LR.values),
    f1_score(df.actual_label.values, df.predicted_LR.values)
)

# ------------------- результати -------------------

print("Accuracy RF:", liashenko_accuracy_score(df.actual_label.values, df.predicted_RF.values))
print("Recall RF:", liashenko_recall_score(df.actual_label.values, df.predicted_RF.values))
print("Precision RF:", liashenko_precision_score(df.actual_label.values, df.predicted_RF.values))
print("F1 RF:", liashenko_f1_score(df.actual_label.values, df.predicted_RF.values))
print("")
print("Accuracy LR:", liashenko_accuracy_score(df.actual_label.values, df.predicted_LR.values))
print("Recall LR:", liashenko_recall_score(df.actual_label.values, df.predicted_LR.values))
print("Precision LR:", liashenko_precision_score(df.actual_label.values, df.predicted_LR.values))
print("F1 LR:", liashenko_f1_score(df.actual_label.values, df.predicted_LR.values))

# ------------------- ROC-крива -------------------

fpr_RF, tpr_RF, _ = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, _ = roc_curve(df.actual_label.values, df.model_LR.values)

auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)

plt.plot(fpr_RF, tpr_RF, "r-", label="RF AUC: %.3f" % auc_RF)
plt.plot(fpr_LR, tpr_LR, "b-", label="LR AUC: %.3f" % auc_LR)
plt.plot([0, 1], [0, 1], "k--", label="random")
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], "g--", label="perfect")
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve")
plt.show()
