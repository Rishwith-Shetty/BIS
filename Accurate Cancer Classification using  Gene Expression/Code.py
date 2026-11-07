!pip install pandas numpy scikit-learn matplotlib seaborn

from google.colab import files

print("Upload data.csv")
uploaded_data = files.upload()

print("Upload labels.csv")
uploaded_labels = files.upload()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# STEP 1: Load dataset
data = pd.read_csv("data.csv")
labels = pd.read_csv("labels.csv")

# STEP 2: Prepare gene expression data
# Match samples using their IDs when available

id_col = data.columns[0]
label_col = next(
    (c for c in labels.columns
     if labels[c].astype(str).str.upper().isin(
         ["BRCA", "KIRC", "COAD", "LUAD", "PRAD"]
     ).any()),
    None
)

if label_col is None:
    raise ValueError("Cancer label column not found")

label_ids = [
    c for c in labels.columns
    if c != label_col and c in data.columns
]

if label_ids:
    df = data.merge(
        labels[[label_ids[0], label_col]],
        left_on=id_col,
        right_on=label_ids[0]
    )
else:
    df = data.copy()
    if len(data) != len(labels):
        raise ValueError("Sample counts do not match")
    df[label_col] = labels[label_col].values

y = df[label_col].astype(str)
X = df.drop(columns=[label_col])

# Remove sample ID columns
X = X.drop(
    columns=[c for c in X.columns
             if c.lower().startswith("unnamed")
             or c == id_col],
    errors="ignore"
)

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median()).fillna(0)

print("Dataset Shape:", X.shape)
print("Cancer Types:", y.unique())

# STEP 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# STEP 4: Remove low variance genes
selector = VarianceThreshold(threshold=0.01)

X_train = selector.fit_transform(X_train)
X_test = selector.transform(X_test)

# STEP 5: Normalize gene expression values
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# STEP 6: Apply PCA
n_components = min(
    100,
    X_train.shape[0] - 1,
    X_train.shape[1]
)

pca = PCA(n_components=n_components)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print("PCA Components:", n_components)

# STEP 7: Train classification model
model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# STEP 8: Predict cancer type
y_pred = model.predict(X_test)

# STEP 9: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nCANCER CLASSIFICATION RESULTS")
print("--------------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# STEP 10: Confusion Matrix
cm = confusion_matrix(
    y_test, y_pred,
    labels=model.classes_
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title("Cancer Classification Confusion Matrix")
plt.xlabel("Predicted Cancer Type")
plt.ylabel("Actual Cancer Type")
plt.show()

