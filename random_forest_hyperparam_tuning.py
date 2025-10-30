# random_forest_hyperparam_tuning.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
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

# 3. Define Random Forest model
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# 4. Define parameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# 5. Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 6. Grid Search with cross-validation
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=2
)

print("Starting hyperparameter tuning...")
grid_search.fit(X, y)

# 7. Display best parameters and score
print("\n=== Best Hyperparameters ===")
print(grid_search.best_params_)
print(f"Best ROC-AUC: {grid_search.best_score_:.4f}")

# 8. Train final model using best parameters
best_rf = grid_search.best_estimator_
best_rf.fit(X, y)

# 9. Evaluate via cross-validation predictions
y_pred = cross_val_predict(best_rf, X, y, cv=cv, method='predict')
y_proba = cross_val_predict(best_rf, X, y, cv=cv, method='predict_proba')[:, 1]

acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
auc = roc_auc_score(y, y_proba)

print("\n=== Cross-Validated Performance with Best Model ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y, y_pred))

# 10. Save predictions
predictions_df = pd.DataFrame({
    "Actual": y,
    "Predicted": y_pred,
    "Pred_Prob": y_proba
})
predictions_df.to_csv("rf_best_model_predictions.csv", index=False)
print("\nPredictions saved to: rf_best_model_predictions.csv")

# 11. Feature importances from best model
importances = pd.Series(best_rf.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(10,6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 15 Feature Importances - Tuned Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
