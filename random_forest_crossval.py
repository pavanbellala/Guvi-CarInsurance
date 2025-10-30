# random_forest_crossval.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
data = pd.read_csv("./train_label_encoded.csv")

# 2. Identify target and features
target_col = "is_claim"
X = data.drop(columns=[target_col])
y = data[target_col]

# 3. Define the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# 4. Set up 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 5. Define scoring metrics
scoring = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
    "f1": make_scorer(f1_score),
    "roc_auc": make_scorer(roc_auc_score, needs_proba=True)
}

# 6. Perform cross-validation
cv_results = cross_validate(
    rf_model, X, y, cv=cv, scoring=scoring, return_train_score=False
)

# 7. Display mean and std deviation of metrics
print("=== Random Forest Cross-Validation Results (5-Fold) ===")
for metric in scoring.keys():
    mean_score = cv_results[f'test_{metric}'].mean()
    std_score = cv_results[f'test_{metric}'].std()
    print(f"{metric.capitalize():<10}: {mean_score:.4f} ± {std_score:.4f}")

# 8. Train final model on full dataset
rf_model.fit(X, y)

# 9. Plot top feature importances
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(10,6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 15 Feature Importances - Random Forest (Full Training Data)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
