# random_forest_full_train.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
data = pd.read_csv("./train_label_encoded.csv")

# 2. Identify target and features
target_col = "is_claim"
X = data.drop(columns=[target_col])
y = data[target_col]

# 3. Train Random Forest model on full dataset
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X, y)

# 4. Predict on same data (for baseline training metrics)
y_pred = rf_model.predict(X)
y_pred_proba = rf_model.predict_proba(X)[:, 1]

# 5. Evaluate model performance (on training data)
print("=== Random Forest Baseline (Full Training Data) ===")
print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
print(f"Precision: {precision_score(y, y_pred):.4f}")
print(f"Recall:    {recall_score(y, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y, y_pred))

# 6. Plot feature importance
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(10,6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 15 Feature Importances - Random Forest (Full Training Data)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
