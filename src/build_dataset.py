"""Build data/dataset.csv from downloaded CSV files (no scraping needed).

Basketball-Reference blocks scripts (HTTP 403), so download the free Kaggle dataset
"NBA Stats (1947-present)" by Sumit Rodatta, which is built from Basketball-Reference,
and put these four files in data/external/ (names are matched loosely):

    Player Per Game.csv
    Advanced.csv
    Player Award Shares.csv
    Team Summaries.csv

Usage:
    python src/build_dataset.py --dir data/external --start 1980 --end 2025
"""
import argparse
import re
from pathlib import Path

import pandas as pd


def norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower().replace("%", "_percent")).strip("_")


def read_csv(folder: Path, must_contain: str, must_not=()) -> pd.DataFrame:
    for f in sorted(folder.glob("*.csv")):
        n = f.name.lower().replace("_", " ")
        if must_contain in n and not any(bad in n for bad in must_not):
            df = pd.read_csv(f)
            df.columns = [norm(c) for c in df.columns]
            print(f"  using {f.name}  ({len(df)} rows)")
            return df
    raise SystemExit(f"Could not find a CSV containing '{must_contain}' in {folder}")


def pick(df: pd.DataFrame, *aliases: str) -> str:
    for a in aliases:
        if a in df.columns:
            return a
    raise SystemExit(f"None of the columns {aliases} found. Columns present: {list(df.columns)}")


def one_row_per_player_season(df: pd.DataFrame, tm_col: str, key_cols: list) -> pd.DataFrame:
    """Traded players have a TOT row plus one row per team: keep TOT, remember last team."""
    last_team = (df[df[tm_col] != "TOT"].groupby(key_cols)[tm_col].last().rename("Team"))
    keep = df[(df[tm_col] == "TOT") | (~df.duplicated(key_cols, keep=False))].copy()
    keep = keep.join(last_team, on=key_cols)
    keep["Team"] = keep["Team"].fillna(keep[tm_col])
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/external")
    ap.add_argument("--start", type=int, default=1980)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--out", default="data/dataset.csv")
    args = ap.parse_args()
    folder = Path(args.dir)

    print("Reading files ...")
    pg = read_csv(folder, "per game")
    adv = read_csv(folder, "advanced")
    aw = read_csv(folder, "award")
    tm = read_csv(folder, "team summar")

    season, player, team = pick(pg, "season"), pick(pg, "player"), pick(pg, "tm", "team")
    pg = pg[pg[pick(pg, "lg", "league")] == "NBA"] if "lg" in pg.columns else pg
    pg = pg[(pg[season] >= args.start) & (pg[season] <= args.end)]
    key = [season, player]
    pg = one_row_per_player_season(pg, team, key)

    out = pd.DataFrame({
        "Season": pg[season].astype(int), "Player": pg[player].str.replace("*", "", regex=False).str.strip(),
        "Team": pg["Team"], "G": pg[pick(pg, "g")],
        "MP": pg[pick(pg, "mp_per_game", "mp")], "PTS": pg[pick(pg, "pts_per_game", "pts")],
        "TRB": pg[pick(pg, "trb_per_game", "trb")], "AST": pg[pick(pg, "ast_per_game", "ast")],
        "STL": pg[pick(pg, "stl_per_game", "stl")], "BLK": pg[pick(pg, "blk_per_game", "blk")],
        "TOV": pg[pick(pg, "tov_per_game", "tov")], "FG%": pg[pick(pg, "fg_percent")],
        "3P%": pg[pick(pg, "x3p_percent", "3p_percent")], "FT%": pg[pick(pg, "ft_percent")],
    })

    # advanced stats
    a_season, a_player, a_tm = pick(adv, "season"), pick(adv, "player"), pick(adv, "tm", "team")
    adv = adv[(adv[a_season] >= args.start) & (adv[a_season] <= args.end)]
    if "lg" in adv.columns:
        adv = adv[adv["lg"].astype(str).str.upper() == "NBA"]
    adv = one_row_per_player_season(adv, a_tm, [a_season, a_player])
    adv_out = pd.DataFrame({
        "Season": adv[a_season].astype(int), "Player": adv[a_player].str.replace("*", "", regex=False).str.strip(),
        "PER": adv[pick(adv, "per")], "TS%": adv[pick(adv, "ts_percent")],
        "USG%": adv[pick(adv, "usg_percent")], "WS": adv[pick(adv, "ws")],
        "WS/48": adv[pick(adv, "ws_48")], "BPM": adv[pick(adv, "bpm")], "VORP": adv[pick(adv, "vorp")],
    }).drop_duplicates(["Season", "Player"])
    out = out.merge(adv_out, on=["Season", "Player"], how="left")

    # team win %
    t_season, t_abbr = pick(tm, "season"), pick(tm, "abbreviation", "abbr", "tm")
    tm = tm[tm[t_abbr].notna() & (tm[t_abbr] != "NA")]
    if "lg" in tm.columns:
        tm = tm[tm["lg"].astype(str).str.upper() == "NBA"]
    team_out = pd.DataFrame({
        "Season": tm[t_season].astype(int), "Team": tm[t_abbr],
        "TeamWinPct": tm[pick(tm, "w")] / (tm[pick(tm, "w")] + tm[pick(tm, "l")]),
    }).drop_duplicates(["Season", "Team"])
    out = out.merge(team_out, on=["Season", "Team"], how="left")

    # MVP vote share (target)
    award_col = pick(aw, "award")
    aw = aw[aw[award_col].astype(str).str.lower().str.strip() == "nba mvp"]
    share = pd.DataFrame({
        "Season": aw[pick(aw, "season")].astype(int),
        "Player": aw[pick(aw, "player")].str.replace("*", "", regex=False).str.strip(),
        "Share": aw[pick(aw, "share")],
    }).drop_duplicates(["Season", "Player"])
    out = out.merge(share, on=["Season", "Player"], how="left")
    out["Share"] = out["Share"].fillna(0.0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    voted = out[out["Share"] > 0]
    print(f"\nSaved {len(out)} rows, {out['Season'].nunique()} seasons -> {args.out}")
    print(f"Players with MVP votes: {len(voted)} (should be roughly 10-15 per season)")
    print(f"Missing TeamWinPct: {out['TeamWinPct'].isna().mean():.1%}   Missing PER: {out['PER'].isna().mean():.1%}")
    print("\nLast 5 MVPs found (check these against reality):")
    print(out.loc[out.groupby("Season")["Share"].idxmax()][["Season", "Player", "Share"]].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
