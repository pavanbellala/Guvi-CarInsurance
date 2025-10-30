# predict_claims_random_forest.py
import pandas as pd
import joblib

# 1. Load trained model
model_path = "final_random_forest_model.pkl"
rf_model = joblib.load(model_path)
print(f"Loaded model from: {model_path}")

# 2. Load new data for prediction
new_data = pd.read_csv("./test.csv")

# 3. Ensure feature alignment
train_data = pd.read_csv("./train_label_encoded.csv")
target_col = "is_claim"
X_train = train_data.drop(columns=[target_col])

# Reorder and align columns
new_data = new_data[X_train.columns]

# 4. Predict claim probabilities and labels
new_data["Pred_Prob"] = rf_model.predict_proba(new_data)[:, 1]
new_data["Predicted"] = rf_model.predict(new_data)

# 5. Save predictions to CSV
output_file = "rf_claim_predictions.csv"
new_data.to_csv(output_file, index=False)
print(f"Predictions saved to: {output_file}")

# 6. Display quick summary
print("\n=== Sample Predictions ===")
print(new_data[["Predicted", "Pred_Prob"]].head())
