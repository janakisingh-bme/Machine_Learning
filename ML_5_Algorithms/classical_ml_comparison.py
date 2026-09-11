# Classical Machine Learning — 5 Algorithm Comparison
# Algorithms:
# 1. Logistic Regression
# 2. KNN
# 3. Naive Bayes
# 4. Decision Tree
# 5. SVM

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ---------------------------------------------------------
# 1. Create output folder
# ---------------------------------------------------------

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------

data = load_breast_cancer()

X = data.data
y = data.target

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Number of samples : {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Classes           : {data.target_names}")


# ---------------------------------------------------------
# 3. Train-test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# ---------------------------------------------------------
# 4. Define five ML models
# ---------------------------------------------------------

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Naive Bayes": GaussianNB(),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf"))
    ])
}


# ---------------------------------------------------------
# 5. Train and evaluate models
# ---------------------------------------------------------

results = []

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    # Print metrics
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=data.target_names
        )
    )

    # -----------------------------------------------------
    # Confusion Matrix
    # -----------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))

    plt.imshow(cm, interpolation="nearest")

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(
        range(len(data.target_names)),
        data.target_names
    )

    plt.yticks(
        range(len(data.target_names)),
        data.target_names
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.colorbar()
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{filename}_confusion_matrix.png"
        ),
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# 6. Create comparison table
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# ---------------------------------------------------------
# 7. Save results as CSV
# ---------------------------------------------------------

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    ),
    index=False
)


# ---------------------------------------------------------
# 8. Plot model comparison
# ---------------------------------------------------------

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

x = np.arange(len(results_df["Model"]))
width = 0.18

plt.figure(figsize=(12, 6))

for i, metric in enumerate(metrics):

    plt.bar(
        x + i * width,
        results_df[metric],
        width,
        label=metric
    )

plt.xticks(
    x + width * 1.5,
    results_df["Model"],
    rotation=20
)

plt.ylabel("Score")
plt.title("Classical Machine Learning Model Comparison")

plt.ylim(0, 1.05)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.png"
    ),
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 9. Best model
# ---------------------------------------------------------

best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    f"Best model based on F1 Score: "
    f"{best_model['Model']}"
)

print(
    f"F1 Score: "
    f"{best_model['F1 Score']:.4f}"
)

print("\nResults saved inside:", OUTPUT_DIR)
