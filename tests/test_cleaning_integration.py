# tests/test_cleaning_integration.py

import sys
import os
import pandas as pd
from pathlib import Path

# Make pipeline/src importable
sys.path.append(os.path.abspath("pipeline/src"))

from data_cleaning import main


def test_full_pipeline(tmp_path):
    # create temp raw folder
    raw_dir = tmp_path / "data_raw"
    raw_dir.mkdir()

    clean_dir = tmp_path / "data_clean"
    clean_dir.mkdir()

    # small valid CSV for testing
    df = pd.DataFrame({
        "gameid": [1] * 10,
        "participantid": list(range(10)),
        "side": ["Blue"] * 5 + ["Red"] * 5,
        "position": ["Top","Jng","Mid","Bot","Sup"]*2,
        "champion": ["Aatrox"] * 10,
        "result": [1] * 10,
        "teamname": ["A"]*5 + ["B"]*5,
        "teamid": [100]*5 + [200]*5,
        "league": ["LCK"] * 10,
        "year": [2021] * 10,
        "patch": ["11.1"] * 10,
        "gamelength": [1800] * 10,
        "kills": [2] * 10
    })

    df.to_csv(raw_dir / "sample.csv", index=False)

    cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        main()
    finally:
        os.chdir(cwd)

    assert (clean_dir / "players_clean.csv").exists()
    assert (clean_dir / "teams_clean.csv").exists()
    assert (clean_dir / "matches_clean.csv").exists()
