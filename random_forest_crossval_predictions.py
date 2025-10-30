# random_forest_crossval_predictions.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
data = pd.read_csv("./train_label_encoded.csv")

# 2. Identify target and features
target_col = "is_claim"
X = data.drop(columns=[target_col])
y = data[target_col]

# 3. Define Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# 4. Setup Stratified K-Fold CV
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 5. Initialize containers for metrics and predictions
fold_results = []
all_predictions = []

# 6. Cross-validation loop
for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]
    
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_val)
    y_proba = rf_model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics for this fold
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)
    
    fold_results.append({
        "Fold": fold,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": auc
    })
    
    # Store predictions
    fold_pred_df = pd.DataFrame({
        "Fold": fold,
        "Actual": y_val.values,
        "Predicted": y_pred,
        "Pred_Prob": y_proba
    })
    all_predictions.append(fold_pred_df)

# 7. Combine all folds’ predictions
predictions_df = pd.concat(all_predictions, ignore_index=True)
metrics_df = pd.DataFrame(fold_results)

# 8. Print cross-validation results
print("\n=== Random Forest Cross-Validation Results (5-Fold) ===")
print(metrics_df)
print("\nMean ± Std Dev across folds:")
print(metrics_df.describe().loc[['mean', 'std']])

# 9. Save predictions to CSV
predictions_df.to_csv("rf_crossval_predictions.csv", index=False)
print("\nPredictions saved to: rf_crossval_predictions.csv")

# 10. Train final model on full dataset (for feature importances)
rf_model.fit(X, y)

# 11. Plot feature importances
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(10,6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 15 Feature Importances - Random Forest (Full Training Data)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
