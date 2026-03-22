# =========================
# 1) Import Libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# =========================
# 2) Load Dataset
# =========================
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# =========================
# 3) Explore Dataset
# =========================
print("Shape of X:", X.shape)
print("Number of classes:", len(np.unique(y)))

print("\nClass Distribution:")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Class {class_names[u]}: {c}")

# =========================
# 4) Split Dataset
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 5) Function to Train & Evaluate Models
# =========================
results = []

def evaluate_model(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{name} Confusion Matrix")
    plt.show()

    report = classification_report(y_test, y_pred, output_dict=True)
    results.append([
        name,
        report["macro avg"]["precision"],
        report["macro avg"]["recall"],
        report["macro avg"]["f1-score"],
        acc
    ])

# =========================
# 6) Train Models
# =========================
evaluate_model(KNeighborsClassifier(n_neighbors=3), "KNN")
evaluate_model(SVC(), "SVM")
evaluate_model(LogisticRegression(max_iter=200), "Logistic Regression")
evaluate_model(DecisionTreeClassifier(), "Decision Tree")

# =========================
# 7) Comparison Table
# =========================
df_results = pd.DataFrame(results, columns=[
    "Model", "Precision (Macro)", "Recall (Macro)", "F1-score (Macro)", "Accuracy"
])

print("\n=== Model Comparison Table ===")
print(df_results)

# =========================
# 8) Analysis
# =========================
analysis = """
Analysis:
- Setosa is always perfectly classified.
- Versicolor and Virginica sometimes misclassified due to similar features.
- The best model (likely SVM or KNN) handles non-linear boundaries and generalizes well.
"""
print(analysis)
