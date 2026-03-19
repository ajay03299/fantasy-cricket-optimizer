import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("output/baseline_model_v1_predictions.csv")

y_true = df["fantasy_points_v1"]

baselines = {
    "random_forest_model": df["predicted_fantasy_points_v1"],
    "points_last_1": df["points_last_1"],
    "points_last_3_avg": df["points_last_3_avg"],
    "points_last_5_avg": df["points_last_5_avg"],
    "career_avg_points_before": df["career_avg_points_before"],
}

print("Baseline comparison:\n")

for name, preds in baselines.items():
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    print(f"{name}:")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}\n")