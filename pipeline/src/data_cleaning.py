"""
This script is the Data Cleaning module

This script performs the following:

1. Load all raw CSVs from data_raw/.
2. Standardize column names.
3. Convert datatypes to {numeric, datetime, categorical}.
4. Validate data integrity 
    - no duplicate (gameid, participantid) rows
    - exactly 10 players per match
    - side must be {blue, red}
    - position must be {top, jng, mid, bot, sup}
5. Build team-level dataset (one row per team per game).
6. Build match-level dataset (one row per match).
7. Save results into data_clean/.

To run:
    python data_cleaning.py
"""
import pandas as pd
import numpy as np
import glob
import os


# -----------------------------------------------------------
# 1. LOAD RAW CSVs
# -----------------------------------------------------------

def load_raw_csvs(path="data_raw"):
    """
    This function loads all raw CSV files under the given folder and combines them.

    Parameters
    ----------
    path : str
        Path to the folder containing the raw Oracle's Elixir CSV files.

    Returns
    -------
    pd.DataFrame
        A single dataframe containing all rows from every CSV file found.

    Notes
    -----
    - *All* files ending with .csv are loaded.
    - The function asserts that the folder is not empty.
    """
    files = glob.glob(os.path.join(path, "*.csv"))
    assert len(files) > 0, 'Empty path'

    # Concatenate all CSVs into one dataframe
    df = pd.concat((pd.read_csv(f, low_memory=False) for f in files), ignore_index=True)
    return df


# -----------------------------------------------------------
# 2. COLUMN NAME FIXING
# -----------------------------------------------------------

def standardize_columns(df):
    """
    This function converts all column names into lowercase snake_case.

    Returns
    -------
    pd.DataFrame
        The dataframe with standardized column names.
    """

    df.columns = (
        df.columns
        .str.lower()
        .str.replace('[^0-9a-z]+', '_', regex=True)
        .str.strip('_')
    )
    return df

# -----------------------------------------------------------
# 3. DTYPE CLEANING
# -----------------------------------------------------------

def convert_dtypes(df):
    """
    This function converts datatypes where appropriate.

    - Converts the 'date' column into datetime.
    - Converts numeric-looking fields into numeric types.
    - Normalizes text fields such as side, position, and champion names.

    Returns
    -------
    pd.DataFrame
        The dataframe with updated datatypes.
    """

    # Convert date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Numeric conversion
    for col in df.columns:
        if col in ["playername", "teamname", "champion", "position", "side", "league", "url", "date"]:
            continue
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            # If conversion fails, leave the column as-is
            pass


    # Normalize strings
    if "side" in df.columns:
        df["side"] = df["side"].str.lower().str.strip()

    if "position" in df.columns:
        df["position"] = df["position"].str.lower().str.strip()

    if "champion" in df.columns:
        df["champion"] = (
            df["champion"]
            .astype(str)
            .str.replace("’", "'", regex=False)
            .str.strip()
        )

    return df

# -----------------------------------------------------------
# 4. FILTER REAL PLAYER
# -----------------------------------------------------------  
def filter_real_players(df):
    """
    This function filters out non-player entries present in Oracle's Elixir.

    A real player must:
    - have a valid position in {top, jng, mid, bot, sup}
    - have a valid side in {blue, red}
    - have a non-null champion pick
    """
    valid_positions = {"top", "jng", "mid", "bot", "sup"}
    valid_sides = {"blue", "red"}

    df = df[df["position"].isin(valid_positions)]
    df = df[df["side"].isin(valid_sides)]
    df = df[df["champion"].notna()]
    df = df[df["champion"] != ""]

    return df

# -----------------------------------------------------------
# 5. INTEGRITY CHECKS
# -----------------------------------------------------------

def validate_integrity(df):
    """
    This function checks whether the dataset meets expected structural rules.

    Checks include:
    - Required columns must exist.
    - No duplicate (gameid, participantid) rows.
    - Each match must contain exactly 10 players.
    - side must be one of {blue, red}.
    - position must be one of {top, jng, mid, bot, sup}.

    Returns
    -------
    bool
        True if all checks pass.
    """
    required_cols = ["gameid", "participantid", "side", "position", "result"]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Ensure unique player rows
    assert df[['gameid', 'participantid']].drop_duplicates().shape[0] == df.shape[0], \
        "Duplicate (gameid, participantid) pairs found."

    # Exactly 10 players per game
    counts = df.groupby("gameid").size()
    if not (counts == 10).all():
        bad = counts[counts != 10]
        raise ValueError(f"Some games do not have 10 players:\n{bad}")

    # Check valid values
    valid_sides = {"blue", "red"}
    assert set(df["side"].dropna().unique()).issubset(valid_sides), \
        f"Invalid sides detected: {df['side'].unique()}"

    valid_positions = {"top", "jng", "mid", "bot", "sup"}
    assert set(df["position"].dropna().unique()).issubset(valid_positions), \
        f"Invalid positions detected: {df['position'].unique()}"

    return True


# -----------------------------------------------------------
# 6. TEAM-LEVEL AGGREGATION
# -----------------------------------------------------------

def create_team_dataset(df):
    """
    This function aggregates player-level rows into team-level rows.

    It performs:
    - Summation of numeric statistics per team.
    - Selection of static information such as year, patch, and teamname.
    - Pivoting champion picks by role (top, jng, mid, bot, sup).

    Returns
    -------
    pd.DataFrame
        A dataframe where each row represents one team in one game.
    """

    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in ["gameid", "teamid", "side", "result"]
    ]

    # Sum stats per team
    team = df.groupby(["gameid", "teamid", "side"]).agg({
        **{col: "sum" for col in numeric_cols if col not in ["result"]},
        "result": "max",
        "teamname": "first",
        "league": "first",
        "year": "first",
        "patch": "first",
        "gamelength": "first",
    }).reset_index()

    # Champion picks by position
    picks = df.pivot_table(
        index=["gameid", "teamid", "side"],
        columns="position",
        values="champion",
        aggfunc="first"
    ).reset_index()

    # Flatten column names
    picks.columns = [c if isinstance(c, str) else c[1] for c in picks.columns]

    team = team.merge(picks, on=["gameid", "teamid", "side"], how="left")

    return team


# -----------------------------------------------------------
# 7. MATCH-LEVEL DATASET
# -----------------------------------------------------------

def create_match_dataset(team_df):
    """
    This function converts team-level data into match-level data.

    It pivots the team dataframe so each match becomes one row with:
    - All blue side columns
    - All red side columns
    - Match-level outcome fields

    Returns
    -------
    pd.DataFrame
        A dataframe with one row per match.
    """
    match = team_df.pivot(index="gameid", columns="side")
    match.columns = [f"{c[0]}_{c[1]}" for c in match.columns]
    return match.reset_index()


# -----------------------------------------------------------
# 8. SAVE CLEANED DATA
# -----------------------------------------------------------

def save_outputs(players, teams, matches, path="data_clean"):
    """
    This function saves the cleaned datasets to CSV files.

    Parameters
    ----------
    players : pd.DataFrame
        Player-level cleaned dataset.
    teams : pd.DataFrame
        Team-level cleaned dataset.
    matches : pd.DataFrame
        Match-level cleaned dataset.
    path : str
        Output folder where the CSV files will be written.
    """
    os.makedirs(path, exist_ok=True)
    players.to_csv(os.path.join(path, "players_clean.csv"), index=False)
    teams.to_csv(os.path.join(path, "teams_clean.csv"), index=False)
    matches.to_csv(os.path.join(path, "matches_clean.csv"), index=False)
    print(f"Cleaned datasets saved at {path}/")

# -----------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------

def main():
    """
    This function runs the full data cleaning pipeline from start to finish.

    Steps:
    - load raw CSV files
    - standardize columns
    - convert datatypes
    - validate dataset structure
    - build team and match datasets
    - save all cleaned CSV files
    """
    print("Loading raw CSVs...")
    df = load_raw_csvs()

    print("Standardizing columns...")
    df = standardize_columns(df)

    print("Converting datatypes...")
    df = convert_dtypes(df)

    print("Filtering to real player rows...")
    df = filter_real_players(df)

    print("Running integrity checks...")
    validate_integrity(df)

    print("Building team-level dataset...")
    team_df = create_team_dataset(df)

    print("Building match-level dataset...")
    match_df = create_match_dataset(team_df)

    print("Saving outputs...")
    save_outputs(df, team_df, match_df)

    print("Done.")


if __name__ == "__main__":
    main()
