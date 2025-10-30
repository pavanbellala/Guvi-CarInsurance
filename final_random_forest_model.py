# final_random_forest_model.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
data = pd.read_csv("./train_label_encoded.csv")

# 2. Define target and features
target_col = "is_claim"
X = data.drop(columns=[target_col])
y = data[target_col]

# 3. Use best hyperparameters from tuning step
best_params = {
    'n_estimators': 300,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

# 4. Train final model on full dataset
final_rf = RandomForestClassifier(**best_params)
final_rf.fit(X, y)

# 5. Evaluate model (on full data just for baseline)
y_pred = final_rf.predict(X)
y_proba = final_rf.predict_proba(X)[:, 1]

print("\n=== Final Random Forest Model Performance ===")
print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
print(f"Precision: {precision_score(y, y_pred):.4f}")
print(f"Recall:    {recall_score(y, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y, y_pred))

# 6. Save model for deployment
joblib.dump(final_rf, "final_random_forest_model.pkl")
print("\n✅ Model saved as: final_random_forest_model.pkl")

# 7. Save feature importances
importances = pd.Series(final_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
importances.to_csv("feature_importance.csv", header=["importance"])
print("✅ Feature importances saved to: feature_importance.csv")

# 8. Visualize top 15 features
plt.figure(figsize=(10,6))
sns.barplot(x=importances.head(15).values, y=importances.head(15).index)
plt.title("Top 15 Feature Importances - Final Random Forest Model")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
