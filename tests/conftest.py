# tests/conftest.py
import pandas as pd
import pytest

@pytest.fixture
def sample_raw_df():
    """
    Returns a small synthetic dataframe mimicking Oracle's Elixir structure.
    Used for unit tests.
    """
    df = pd.DataFrame({
        "gameid": [1]*10,
        "participantid": list(range(10)),
        "side": ["Blue"]*5 + ["Red"]*5,
        "position": ["Top","Jng","Mid","Bot","Sup"]*2,
        "champion": ["Aatrox"]*10,
        "result": [1]*10,
        "teamname": ["TeamA"]*5 + ["TeamB"]*5,
        "teamid": [100]*5 + [200]*5,
        "league": ["LCK"]*10,
        "year": [2021]*10,
        "patch": ["11.1"]*10,
        "gamelength": [1800]*10,
        "kills": [2]*10,
        "goldat10": [3000]*10
    })
    return df
