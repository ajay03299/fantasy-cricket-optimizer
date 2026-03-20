from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT_PATH = Path("output/model_feature_table_v3.csv")
PREDICTIONS_PATH = Path("output/best_model_v3_predictions.csv")


def main():
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values(["match_date", "match_id", "player_name"]).reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    target_col = "fantasy_points_v3"
    feature_cols = [c for c in df.columns if c not in ["match_id", "match_date", "player_name", "team", target_col]]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    categorical_features = ["season", "venue", "player_role_platform", "opponent"]
    numeric_features = [c for c in feature_cols if c not in categorical_features]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = GradientBoostingRegressor(
        random_state=42,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print("Train rows:", len(train_df))
    print("Test rows:", len(test_df))
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))

    out = test_df.copy()
    out["predicted_fantasy_points_v3"] = preds
    out.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Predictions saved to: {PREDICTIONS_PATH}")

    preview_cols = [
        "match_date",
        "player_name",
        "team",
        "opponent",
        "player_role_platform",
        "fantasy_points_v3",
        "predicted_fantasy_points_v3",
    ]
    print("\nSample predictions:")
    print(out[preview_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()