"""Shared data loading and leakage-safe feature engineering for NBA MVP prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "PTS", "TRB", "AST", "STL", "BLK", "TOV", "FG%", "3P%", "FT%", "MP", "G",
    "PER", "TS%", "USG%", "WS", "WS/48", "BPM", "VORP", "TeamWinPct",
]
TARGET = "Share"
MIN_GAMES = 41
CANDIDATES_PER_SEASON = 30


def load_dataset(path: str = "data/dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Season", "Player", TARGET, *FEATURES}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    for col in FEATURES + [TARGET, "Season", "G"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Season", "Player", TARGET, "G"])
    df["Season"] = df["Season"].astype(int)
    return df


def build_candidates(df: pd.DataFrame, candidates_per_season: int = CANDIDATES_PER_SEASON) -> pd.DataFrame:
    """Build an MVP candidate pool without using actual MVP votes to select rows.

    Eligibility uses games played and Win Shares only. Actual vote share is used
    solely as the target. Season-relative percentiles are computed among all
    eligible players before narrowing to the top Win Shares candidates.
    """
    if candidates_per_season < 1:
        raise ValueError("candidates_per_season must be at least 1")
    df = df[df["G"] >= MIN_GAMES].copy()
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    relative_cols = ["PTS", "TRB", "AST", "PER", "WS", "BPM", "VORP"]
    for col in relative_cols:
        df[f"{col}_rel"] = df.groupby("Season")[col].rank(pct=True, method="average")

    # Crucially, do not include (TARGET > 0) here: that would use the answer
    # to decide which rows are eligible for evaluation.
    df["ws_rank"] = df.groupby("Season")["WS"].rank(ascending=False, method="first")
    df = df[df["ws_rank"] <= candidates_per_season].copy()
    return df.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return FEATURES + [c for c in df.columns if c.endswith("_rel")]


def top_k_accuracy(df: pd.DataFrame, pred_col: str, k: int = 1) -> float:
    """Fraction of seasons whose actual MVP is among the top-k predictions."""
    if k < 1:
        raise ValueError("k must be at least 1")
    hits, seasons = 0, 0
    for _, group in df.groupby("Season"):
        if group[TARGET].max() <= 0:
            continue
        true_mvp_idx = group[TARGET].idxmax()
        top_indices = group.nlargest(k, pred_col).index
        hits += int(true_mvp_idx in top_indices)
        seasons += 1
    return hits / seasons if seasons else float("nan")
