"""Predict and display an MVP race for a season in the dataset.

For a historical evaluation/demo, use --honest to exclude that season from
training. The saved model was fit on all seasons and is intended for inference
on a genuinely unseen season, not retrospective evaluation.
"""
import argparse

import joblib
import numpy as np

from features import TARGET, build_candidates, load_dataset
from train import get_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True, help="season-ending year, e.g. 2025 for 2024-25")
    parser.add_argument("--data", default="data/dataset.csv")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--honest", action="store_true", help="retrain excluding the selected season")
    args = parser.parse_args()
    if args.top < 1:
        raise SystemExit("--top must be at least 1")

    df = build_candidates(load_dataset(args.data))
    bundle = joblib.load("models/best_model.joblib")
    cols, name = bundle["columns"], bundle["name"]
    season = df[df["Season"] == args.season].copy()
    if season.empty:
        raise SystemExit(f"No eligible candidate data for season {args.season}")

    if args.honest:
        train = df[df["Season"] != args.season]
        if train.empty:
            raise SystemExit("No training data remains after excluding this season")
        model = get_models()[name]
        model.fit(train[cols], train[TARGET])
    else:
        model = bundle["model"]
        print("Note: saved model was trained on all seasons; use --honest for historical evaluation.")

    season["predicted_share"] = np.clip(model.predict(season[cols]), 0, 1)
    ranked = season.sort_values("predicted_share", ascending=False).head(args.top)
    print(f"\nPredicted MVP race, {args.season - 1}-{str(args.season)[2:]} (model: {name}{', honest' if args.honest else ''})")
    print("-" * 72)
    for rank, (_, row) in enumerate(ranked.iterrows(), 1):
        actual = f" | actual share {row[TARGET]:.3f}" if row[TARGET] > 0 else ""
        print(f"{rank:>2}. {row['Player']:<27} predicted share {row['predicted_share']:.3f}{actual}")
    print(f"\nPredicted MVP: {ranked.iloc[0]['Player']}")


if __name__ == "__main__":
    main()
