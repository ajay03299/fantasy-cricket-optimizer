from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)

INPUT_PATH = Path("output/model_feature_table_v3.csv")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


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

    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # For HistGradientBoosting, use only numeric columns
    X_train_num = train_df[numeric_features].fillna(0)
    X_test_num = test_df[numeric_features].fillna(0)

    models = {
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                ("model", RandomForestRegressor(
                    n_estimators=400,
                    random_state=42,
                    n_jobs=-1,
                    max_depth=16,
                    min_samples_leaf=3,
                )),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                ("model", ExtraTreesRegressor(
                    n_estimators=400,
                    random_state=42,
                    n_jobs=-1,
                    max_depth=16,
                    min_samples_leaf=3,
                )),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                ("model", GradientBoostingRegressor(
                    random_state=42,
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.8,
                )),
            ]
        ),
    }

    results = []

    print("Model comparison on V3 features:\n")

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        test_rmse = rmse(y_test, preds)

        results.append({
            "model": name,
            "mae": mae,
            "rmse": test_rmse,
        })

        print(f"{name}:")
        print(f"  MAE  = {mae:.4f}")
        print(f"  RMSE = {test_rmse:.4f}\n")

    # HistGradientBoosting on numeric-only data
    hgb = HistGradientBoostingRegressor(
        random_state=42,
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=20,
    )
    hgb.fit(X_train_num, y_train)
    preds = hgb.predict(X_test_num)

    mae = mean_absolute_error(y_test, preds)
    test_rmse = rmse(y_test, preds)

    results.append({
        "model": "hist_gradient_boosting_numeric_only",
        "mae": mae,
        "rmse": test_rmse,
    })

    print("hist_gradient_boosting_numeric_only:")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {test_rmse:.4f}\n")

    results_df = pd.DataFrame(results).sort_values("mae")
    print("Sorted results:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()