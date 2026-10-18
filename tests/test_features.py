import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import TARGET, build_candidates, feature_columns


class CandidateSelectionTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "Season": [2020] * 5,
            "Player": ["A", "B", "C", "D", "E"],
            "G": [70, 68, 65, 40, 60],
            "WS": [10, 9, 8, 20, 7],
            "Share": [0.0, 0.9, 0.0, 1.0, 0.0],
            "PTS": [10, 20, 8, 30, 7], "TRB": [1, 2, 3, 4, 5],
            "AST": [1, 2, 3, 4, 5], "STL": [1] * 5, "BLK": [1] * 5,
            "TOV": [1] * 5, "FG%": [0.4] * 5, "3P%": [0.3] * 5,
            "FT%": [0.8] * 5, "MP": [30] * 5, "PER": [10, 20, 8, 30, 7],
            "TS%": [0.5] * 5, "USG%": [20] * 5, "WS/48": [0.1] * 5,
            "BPM": [1, 2, 3, 4, 5], "VORP": [1, 2, 3, 4, 5],
            "TeamWinPct": [0.5] * 5,
        })

    def test_filter_does_not_rescue_player_using_target(self):
        # Player D has the highest actual share but is ineligible by games played.
        candidates = build_candidates(self.df, candidates_per_season=2)
        self.assertNotIn("D", candidates["Player"].tolist())
        self.assertEqual(set(candidates["Player"]), {"A", "B"})

    def test_relative_features_and_columns_exist(self):
        candidates = build_candidates(self.df, candidates_per_season=3)
        self.assertIn("PTS_rel", candidates.columns)
        self.assertIn("VORP_rel", feature_columns(candidates))
        self.assertIn(TARGET, candidates.columns)

    def test_candidate_count_limit(self):
        candidates = build_candidates(self.df, candidates_per_season=2)
        self.assertLessEqual(len(candidates), 2)


if __name__ == "__main__":
    unittest.main()
