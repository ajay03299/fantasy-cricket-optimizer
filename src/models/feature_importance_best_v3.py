from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor

INPUT_PATH = Path("output/model_feature_table_v3.csv")

df = pd.read_csv(INPUT_PATH, low_memory=False)
df["match_date"] = pd.to_datetime(df["match_date"])
df = df.sort_values(["match_date", "match_id", "player_name"]).reset_index(drop=True)

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()

target_col = "fantasy_points_v3"
feature_cols = [c for c in df.columns if c not in ["match_id", "match_date", "player_name", "team", target_col]]

categorical_features = ["season", "venue", "player_role_platform", "opponent"]
numeric_features = [c for c in feature_cols if c not in categorical_features]

X_train = train_df[feature_cols]
y_train = train_df[target_col]

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

feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = pipeline.named_steps["model"].feature_importances_

fi = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("Top 30 feature importances:\n")
print(fi.head(30).to_string(index=False))