from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

INPUT_PATH = Path("output/model_feature_table_v1.csv")

df = pd.read_csv(INPUT_PATH)
df["match_date"] = pd.to_datetime(df["match_date"])
df = df.sort_values(["match_date", "match_id", "player_name"]).reset_index(drop=True)

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()

target_col = "fantasy_points_v1"

feature_cols = [
    "season",
    "venue",
    "player_role_platform",
    "matches_played_before",
    "career_avg_points_before",
    "points_last_1",
    "points_last_3_avg",
    "points_last_5_avg",
    "runs_last_1",
    "runs_last_3_avg",
    "runs_last_5_avg",
    "wickets_last_1",
    "wickets_last_3_avg",
    "wickets_last_5_avg",
    "balls_faced_last_1",
    "balls_faced_last_3_avg",
    "balls_faced_last_5_avg",
    "balls_bowled_last_1",
    "balls_bowled_last_3_avg",
    "balls_bowled_last_5_avg",
    "avg_points_at_venue_before",
    "avg_runs_at_venue_before",
    "avg_points_vs_opponent_before",
    "avg_wickets_vs_opponent_before",
    "opponent",
]

numeric_features = [
    "matches_played_before",
    "career_avg_points_before",
    "points_last_1",
    "points_last_3_avg",
    "points_last_5_avg",
    "runs_last_1",
    "runs_last_3_avg",
    "runs_last_5_avg",
    "wickets_last_1",
    "wickets_last_3_avg",
    "wickets_last_5_avg",
    "balls_faced_last_1",
    "balls_faced_last_3_avg",
    "balls_faced_last_5_avg",
    "balls_bowled_last_1",
    "balls_bowled_last_3_avg",
    "balls_bowled_last_5_avg",
    "avg_points_at_venue_before",
    "avg_runs_at_venue_before",
    "avg_points_vs_opponent_before",
    "avg_wickets_vs_opponent_before",
]

categorical_features = [
    "season",
    "venue",
    "player_role_platform",
    "opponent",
]

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

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_depth=12,
    min_samples_leaf=5,
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