"""
Fix v8 duplicate rows caused by toss merge.
Deduplicates on (match_id, player_name), keeping first occurrence.
"""
import pandas as pd

ft8 = pd.read_csv("output/model_feature_table_v8.csv", low_memory=False)
print(f"Before: {len(ft8)} rows")
print(f"Duplicates: {ft8.duplicated(subset=['match_id', 'player_name']).sum()}")

ft8 = ft8.drop_duplicates(subset=['match_id', 'player_name'], keep='first')
print(f"After:  {len(ft8)} rows")

ft8.to_csv("output/model_feature_table_v8.csv", index=False)
print("Saved clean v8 feature table.")
