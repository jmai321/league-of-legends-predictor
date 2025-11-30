# tests/test_cleaning_unit.py

import sys
import os
import pandas as pd

# Make pipeline/src importable
sys.path.append(os.path.abspath("pipeline/src"))

from data_cleaning import (
    standardize_columns,
    convert_dtypes,
    filter_real_players,
    validate_integrity,
    create_team_dataset,
    create_match_dataset,
)


def test_standardize_columns():
    df = pd.DataFrame(columns=["Team KPM", "GoldAt10", "dragons (type unknown)"])
    df = standardize_columns(df)
    assert df.columns.tolist() == ["team_kpm", "goldat10", "dragons_type_unknown"]


def test_convert_dtypes():
    df = pd.DataFrame({
        "date": ["2021-01-01", "2021-01-02"],
        "kills": ["5", "8"],
        "side": ["Blue", "Red"],
        "position": ["Top", "Mid"],
        "champion": ["Aatrox", "Ahri"],
        "result": [1, 0]
    })

    out = convert_dtypes(df)

    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert out["kills"].dtype in (int, float)
    assert set(out["side"]) == {"blue", "red"}


def test_filter_real_players():
    df = pd.DataFrame({
        "position": ["top", "coach", "mid", ""],
        "side": ["blue", "blue", "red", ""],
        "champion": ["Aatrox", "", "Ahri", None]
    })

    filtered = filter_real_players(df)
    assert len(filtered) == 2
    assert set(filtered["position"]) == {"top", "mid"}


def test_validate_integrity_pass():
    df = pd.DataFrame({
        "gameid": [1] * 10,
        "participantid": list(range(10)),
        "side": ["blue"]*5 + ["red"]*5,
        "position": ["top","jng","mid","bot","sup"]*2,
        "champion": ["Aatrox"]*10,
        "result": [1]*10
    })
    assert validate_integrity(df) is True


def test_validate_integrity_fail():
    df = pd.DataFrame({
        "gameid": [1] * 9,           # only 9 rows -> should fail
        "participantid": list(range(9)),
        "side": ["blue"]*5 + ["red"]*4,
        "position": ["top","jng","mid","bot","sup","top","jng","mid","bot"],
        "champion": ["Aatrox"]*9,
        "result": [1]*9
    })

    try:
        validate_integrity(df)
        assert False   # should not reach here
    except Exception:
        assert True


def test_create_team_dataset():
    df = pd.DataFrame({
        "gameid": [1]*10,
        "participantid": range(10),
        "side": ["blue"]*5 + ["red"]*5,
        "teamid": [100]*5 + [200]*5,
        "teamname": ["A"]*5 + ["B"]*5,
        "position": ["top","jng","mid","bot","sup"]*2,
        "champion": ["Aatrox"]*10,
        "kills": [1]*10,
        "result": [1]*10,
        "league": ["LCK"]*10,
        "year": [2021]*10,
        "patch": ["11.1"]*10,
        "gamelength": [1800]*10
    })

    team = create_team_dataset(df)
    assert len(team) == 2
    assert "top" in team.columns  # champion pivot worked


def test_create_match_dataset():
    df = pd.DataFrame({
        "gameid": [1]*10,
        "participantid": range(10),
        "side": ["blue"]*5 + ["red"]*5,
        "teamid": [100]*5 + [200]*5,
        "teamname": ["A"]*5 + ["B"]*5,
        "position": ["top","jng","mid","bot","sup"]*2,
        "champion": ["Aatrox"]*10,
        "kills": [1]*10,
        "result": [1]*10,
        "league": ["LCK"]*10,
        "year": [2021]*10,
        "patch": ["11.1"]*10,
        "gamelength": [1800]*10
    })

    team_df = create_team_dataset(df)
    match = create_match_dataset(team_df)

    assert len(match) == 1
    assert any("teamname" in col for col in match.columns)
