"""Train and evaluate models for NBA MVP vote-share prediction.

The final seasons are held out as a temporal test set. Candidate selection does
not use MVP vote share; see features.py. Results are written to results/.
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from features import TARGET, build_candidates, feature_columns, load_dataset, top_k_accuracy

RESULTS = Path("results")
MODELS = Path("models")
DEFAULT_TEST_SEASONS = list(range(2016, 2026))


def get_models():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=-1),
    }


def evaluate(name, model, train, test, cols):
    model.fit(train[cols], train[TARGET])
    scored = test.copy()
    scored["pred"] = np.clip(model.predict(scored[cols]), 0, 1)
    metrics = {
        "model": name,
        "rmse": float(np.sqrt(mean_squared_error(scored[TARGET], scored["pred"]))),
        "mae": float(mean_absolute_error(scored[TARGET], scored["pred"])),
        "r2": float(r2_score(scored[TARGET], scored["pred"])),
        "top1_acc": float(top_k_accuracy(scored, "pred", 1)),
        "top3_acc": float(top_k_accuracy(scored, "pred", 3)),
        "n_test_rows": int(len(scored)),
        "n_test_seasons": int(scored["Season"].nunique()),
    }
    return metrics, scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset.csv")
    parser.add_argument("--test-seasons", type=int, nargs="+", default=DEFAULT_TEST_SEASONS,
                        help="season-ending years held out for testing")
    parser.add_argument("--candidates", type=int, default=30,
                        help="top Win Shares players retained per season; must not depend on MVP votes")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(args.data)
    df = build_candidates(raw, candidates_per_season=args.candidates)
    cols = feature_columns(df)
    test_seasons = sorted(set(args.test_seasons))
    train = df[~df["Season"].isin(test_seasons)].copy()
    test = df[df["Season"].isin(test_seasons)].copy()
    if train.empty or test.empty:
        raise SystemExit("Training or test set is empty. Check --test-seasons and dataset range.")
    missing_test = sorted(set(test_seasons) - set(test["Season"]))
    if missing_test:
        print(f"Warning: requested test seasons absent from data: {missing_test}")
    print(f"Raw dataset: {len(raw)} player-seasons, {raw['Season'].nunique()} seasons")
    print(f"Candidate pool: {len(df)} rows, {df['Season'].nunique()} seasons, top {args.candidates} by WS")
    print(f"Train: {len(train)} rows / {train['Season'].nunique()} seasons | Test: {len(test)} rows / {test['Season'].nunique()} seasons | {len(cols)} features")

    rows, scored_models = [], {}
    for name, model in get_models().items():
        metrics, scored = evaluate(name, model, train, test, cols)
        rows.append(metrics)
        scored_models[name] = scored
        print(f"{name:18s} RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} R2={metrics['r2']:.3f} "
              f"top1={metrics['top1_acc']:.2f} top3={metrics['top3_acc']:.2f}")

    # Transparent simple baseline: rank eligible candidates by Win Shares.
    baseline = test.copy()
    baseline["baseline_rank_score"] = baseline["WS"]
    # This baseline is a ranking rule, not a calibrated vote-share regressor;
    # therefore RMSE/MAE/R2 are intentionally not reported for it.
    baseline_metrics = {
        "model": "WinSharesRankBaseline",
        "rmse": None,
        "mae": None,
        "r2": None,
        "top1_acc": float(top_k_accuracy(baseline, "baseline_rank_score", 1)),
        "top3_acc": float(top_k_accuracy(baseline, "baseline_rank_score", 3)),
        "n_test_rows": int(len(baseline)),
        "n_test_seasons": int(baseline["Season"].nunique()),
    }
    rows.append(baseline_metrics)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "model_comparison.csv", index=False)

    model_table = table[table["model"] != "WinSharesRankBaseline"].copy()
    # Primary objective is winner identification; RMSE breaks ties.
    model_table = model_table.sort_values(["top1_acc", "rmse"], ascending=[False, True])
    best_name = str(model_table.iloc[0]["model"])
    print(f"\nSelected model (top-1 accuracy, then RMSE tie-break): {best_name}")

    scored = scored_models[best_name]
    per_season = []
    for season, group in scored.groupby("Season"):
        true_idx = group[TARGET].idxmax()
        pred_idx = group["pred"].idxmax()
        per_season.append({
            "season": int(season),
            "actual_mvp": group.loc[true_idx, "Player"],
            "predicted_mvp": group.loc[pred_idx, "Player"],
            "correct": bool(true_idx == pred_idx),
            "actual_share": float(group.loc[true_idx, TARGET]),
            "predicted_share_for_winner": float(group.loc[pred_idx, "pred"]),
            "candidate_count": int(len(group)),
        })
    pd.DataFrame(per_season).to_csv(RESULTS / "per_season_predictions.csv", index=False)

    # Save test predictions for transparent error analysis.
    scored[["Season", "Player", TARGET, "pred"]].rename(columns={TARGET: "actual_share", "pred": "predicted_share"}).to_csv(
        RESULTS / "test_predictions.csv", index=False)

    final = get_models()[best_name]
    final.fit(df[cols], df[TARGET])
    joblib.dump({"model": final, "columns": cols, "name": best_name,
                 "candidate_count": args.candidates, "min_games": 41}, MODELS / "best_model.joblib")

    # Plot top-k accuracy separately from error metrics for easier interpretation.
    plot_table = table.set_index("model")
    ax = plot_table[["top1_acc", "top3_acc"]].plot.bar(rot=20, figsize=(8, 4))
    ax.set_ylabel("Fraction of held-out seasons")
    ax.set_ylim(0, 1.05)
    ax.set_title("MVP winner within top-k predictions (held-out seasons)")
    plt.tight_layout(); plt.savefig(RESULTS / "model_comparison.png", dpi=160); plt.close()

    error_table = table[table["model"] != "WinSharesRankBaseline"].set_index("model")[["rmse", "mae"]]
    ax = error_table.plot.bar(rot=20, figsize=(8, 4))
    ax.set_ylabel("Vote-share error (lower is better)")
    ax.set_title("Held-out vote-share prediction error")
    plt.tight_layout(); plt.savefig(RESULTS / "error_comparison.png", dpi=160); plt.close()

    inner = final[-1] if hasattr(final, "steps") else final
    if hasattr(inner, "feature_importances_"):
        imp = pd.Series(inner.feature_importances_, index=cols).sort_values().tail(12)
        ax = imp.plot.barh(figsize=(7, 5)); ax.set_title(f"Top features ({best_name})")
        ax.set_xlabel("Feature importance")
        plt.tight_layout(); plt.savefig(RESULTS / "feature_importance.png", dpi=160); plt.close()

    metadata = {
        "target": TARGET,
        "candidate_rule": f"G >= 41 and top {args.candidates} by Win Shares per season; no target-based filtering",
        "test_seasons_requested": test_seasons,
        "test_seasons_present": sorted(int(x) for x in test["Season"].unique()),
        "raw_rows": int(len(raw)), "candidate_rows": int(len(df)),
        "feature_columns": cols, "selected_model": best_name,
        "selection_criterion": "highest top-1 accuracy, then lowest RMSE",
        "metrics": rows,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metadata, indent=2))
    print(f"Results saved to {RESULTS}/ and model to {MODELS}/best_model.joblib")


if __name__ == "__main__":
    main()
