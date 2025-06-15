import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.utils import resample
from sklearn.multiclass import OneVsRestClassifier

# ---------------- Load the dataset ----------------
df = pd.read_csv("data/Personalized_Diet_Recommendations.csv")

# ---------------- Drop unnecessary columns ----------------
drop_cols = [
    'Patient_ID',
    'Recommended_Calories', 'Recommended_Protein',
    'Recommended_Carbs', 'Recommended_Fats'
]
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# ---------------- Drop rows with missing values ----------------
df.dropna(inplace=True)

# ---------------- Encode categorical columns ----------------
categorical_cols = [
    'Gender', 'Chronic_Disease', 'Genetic_Risk_Factor', 'Allergies',
    'Exercise_Frequency', 'Alcohol_Consumption', 'Smoking_Habit',
    'Dietary_Habits', 'Preferred_Cuisine', 'Food_Aversions',
    'Cholesterol_Level', 'Recommended_Meal_Plan'
]

label_encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    else:
        print(f"⚠️ Column not found in dataset: {col}")

# ---------------- Prepare Features and Target ----------------
X = df.drop('Recommended_Meal_Plan', axis=1)
y = df['Recommended_Meal_Plan']

# ---------------- Plot Original Class Distribution ----------------
plt.figure(figsize=(12, 5))
sns.countplot(x=y)
plt.title("Original Class Distribution")
plt.xlabel("Diet Plan (Encoded)")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------- Balance the dataset ----------------
df_combined = pd.concat([X, y], axis=1)
min_count = y.value_counts().min()

df_balanced = pd.concat([
    resample(group, replace=False, n_samples=min_count, random_state=42)
    for _, group in df_combined.groupby('Recommended_Meal_Plan')
])

X = df_balanced.drop('Recommended_Meal_Plan', axis=1)
y = df_balanced['Recommended_Meal_Plan']

# ---------------- Plot Balanced Class Distribution ----------------
plt.figure(figsize=(12, 5))
sns.countplot(x=y)
plt.title("Balanced Class Distribution")
plt.xlabel("Diet Plan (Encoded)")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------- Train/Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- Feature Scaling ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- Train the Model ----------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# ---------------- Evaluate ----------------
preds = model.predict(X_test_scaled)
print("\n✅ Accuracy:", accuracy_score(y_test, preds))
print("✅ Classification Report:\n", classification_report(y_test, preds))

# ---------------- Confusion Matrix ----------------
conf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ---------------- Feature Importances ----------------
importances = model.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# ---------------- Optional: Multi-class ROC Curve ----------------
try:
    y_bin = label_binarize(y, classes=np.unique(y))
    classifier = OneVsRestClassifier(RandomForestClassifier(random_state=42))
    classifier.fit(X_train_scaled, y_train)
    y_score = classifier.predict_proba(X_test_scaled)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_bin.shape[1]

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label_binarize(y_test, classes=np.unique(y))[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title("Multi-class ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"ROC curve plotting failed: {e}")

# ---------------- Save Artifacts ----------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/diet_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

print("✅ Model and preprocessing files saved to 'models/' directory.")
