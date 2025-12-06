"""
This script is the Data Cleaning module.

Steps:
1. Load raw CSV files from data_raw/.
2. Standardize column names.
3. Split into:
       a) player-level rows
       b) team-level rows
4. Build data sets
8. Save results into data_clean/.

To run:
    python pipeline/src/data_cleaning.py
"""

import pandas as pd
import glob
import os


# -----------------------------------------------------------
# load raw csvs
# -----------------------------------------------------------

def load_raw_csvs(path="pipeline/src/data_raw"):
    '''
    Load and combine all CSV files in the given folder.

    Parameters
    ----------
    path : str
        folder containing CSV files

    Returns
    -------
    pd.DataFrame
        combined dataframe
    '''
    files = glob.glob(os.path.join(path, "*.csv"))
    assert len(files) > 0, "no CSV files found"

    df = pd.concat(
        [pd.read_csv(f, low_memory=False) for f in files],
        ignore_index=True
    )
    return df


# -----------------------------------------------------------
# standardize column names
# -----------------------------------------------------------

def standardize_columns(df):
    '''
    Convert all column names to lowercase snake_case.
    Parameters
    ----------
    df : pd.DataFrame
        original dataframe to be standardized
    Returns
    -------
    pd.DataFrame
        dataframe with standardized column names
    '''
    df = df.copy()
    df.columns = (
        df.columns
            .str.lower()
            .str.replace("[^0-9a-z]+", "_", regex=True)
            .str.strip("_")
    )
    return df


# -----------------------------------------------------------
# split into player-level and team-level rows
# -----------------------------------------------------------

def split_player_and_team_rows(df):
    '''
    Split rows into player-level and team-level rows.
    Parameters
    ----------
    df : pd.DataFrame
        dataframe already standardized but needs to be splitted
    Returns
    -------
    (player_rows, team_rows)
    '''
    df = df.copy()
    roles = ["top", "jng", "mid", "bot", "sup"]

    players = df[df["position"].isin(roles)].copy()
    teams   = df[df["position"] == "team"].copy()

    return players, teams


# -----------------------------------------------------------
# remove unwanted columns
# -----------------------------------------------------------

def clean_columns(df, drop_cols):
    '''
    Remove columns listed in drop_cols.
    
    Parameters
    ----------
    df : pd.DataFrame
        dataframe to be processed
    drop_cols : list
        list of columns to be removed
    
    Returns
    -------
    pd.DataFrame
        processed data frame
    '''
    
    df = df.copy()
    drop_cols = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=drop_cols)


# -----------------------------------------------------------
# build game_result dataset (Model A)
# -----------------------------------------------------------

def build_result_dataset(player_rows):
    '''
    Build game result table for each match.
    Parameters
    ----------
    player_rows : pd.DataFrame
        player dataframe (already standardized and splitted) from split_player_and_team_rows()
    Returns
    -------
    pd.DataFrame
    (player_rows, team_rows)
        Output columns:
            gameid
            top_blue ... sup_blue
            top_red  ... sup_red
            win (1 if blue side won)
    '''
    if len(player_rows) == 0:
        return pd.DataFrame()

    pr = player_rows.copy()
    pr["side"] = pr["side"].str.lower().str.strip()
    pr["position"] = pr["position"].str.lower().str.strip()

    roles = ["top", "jng", "mid", "bot", "sup"]
    pr = pr[pr["position"].isin(roles)]
    pr = pr.dropna(subset=["champion"])

    # pivot champion picks into one row per (gameid, side)
    lineup = (
        pr.pivot_table(
            index=["gameid", "side"],
            columns="position",
            values="champion",
            aggfunc="first"
        ).reset_index()
    )

    # flatten pivot columns
    lineup.columns = [
        c if isinstance(c, str) else c[1] for c in lineup.columns
    ]
    lineup = lineup.dropna(subset=roles)

    # split sides
    blue = lineup[lineup["side"] == "blue"].copy()
    red  = lineup[lineup["side"] == "red"].copy()

    merged = blue.merge(red, on="gameid", suffixes=("_blue", "_red"))
    if merged.empty:
        return pd.DataFrame()

    # determine winner from blue side result column
    win_map = (
        pr[pr["side"] == "blue"]
        .groupby("gameid")["result"]
        .max()
    )

    merged["win"] = merged["gameid"].map(win_map)
    merged = merged.dropna(subset=["win"])
    if merged.empty:
        return pd.DataFrame()

    merged["win"] = merged["win"].astype(int)

    # remove leftover side columns
    return clean_columns(merged, ["side_blue", "side_red"])


# -----------------------------------------------------------
# build realtime dataset (Model B)
# -----------------------------------------------------------

def build_realtime_dataset(df, game_result):
    '''
    Merge team-level real-time stats with champion lineups.

    Parameters
    ----------
    df : pd.DataFrame
        team-level rows

    game_result : pd.DataFrame
        from build_result_dataset()

    Returns
    -------
    pd.DataFrame
        Output columns:
            realtime_fields from df
            top,jng,mid,bot,sup  from game_result
            match_win(1: this team wins)
    '''
    realtime_fields = [
        "gameid", "side", "datacompleteness", "teamname", "teamid", "gamelength", "result",
        "kills", "deaths", "assists", "teamkills", "teamdeaths",
        "doublekills", "triplekills", "quadrakills", "pentakills",
        "firstblood", "team_kpm", "ckpm", "firstdragon", "dragons",
        "opp_dragons", "elementaldrakes", "opp_elementaldrakes",
        "infernals", "mountains", "clouds", "oceans", "chemtechs",
        "hextechs", "dragons_type_unknown", "elders", "opp_elders",
        "firstherald", "heralds", "opp_heralds", "firstbaron", "barons",
        "opp_barons", "firsttower", "towers", "opp_towers", "firstmidtower",
        "firsttothreetowers", "turretplates", "opp_turretplates",
        "inhibitors", "opp_inhibitors", "damagetochampions", "dpm",
        "damagetakenperminute", "damagemitigatedperminute",
        "damagetotowers", "wardsplaced", "wpm", "wardskilled", "wcpm",
        "controlwardsbought", "visionscore", "vspm", "totalgold",
        "earnedgold", "earned_gpm", "earnedgoldshare", "goldspent", "gspd",
        "gpr", "minionkills", "monsterkills", "cspm", "goldat10", "xpat10",
        "csat10", "opp_goldat10", "opp_xpat10", "opp_csat10",
        "golddiffat10", "xpdiffat10", "csdiffat10", "killsat10",
        "assistsat10", "deathsat10", "opp_killsat10", "opp_assistsat10",
        "opp_deathsat10", "goldat15", "xpat15", "csat15", "opp_goldat15",
        "opp_xpat15", "opp_csat15", "golddiffat15", "xpdiffat15",
        "csdiffat15", "killsat15", "assistsat15", "deathsat15",
        "opp_killsat15", "opp_assistsat15", "opp_deathsat15", "goldat20",
        "xpat20", "csat20", "opp_goldat20", "opp_xpat20", "opp_csat20",
        "golddiffat20", "xpdiffat20", "csdiffat20", "killsat20",
        "assistsat20", "deathsat20", "opp_killsat20", "opp_assistsat20",
        "opp_deathsat20", "goldat25", "xpat25", "csat25", "opp_goldat25",
        "opp_xpat25", "opp_csat25", "golddiffat25", "xpdiffat25",
        "csdiffat25", "killsat25", "assistsat25", "deathsat25",
        "opp_killsat25", "opp_assistsat25", "opp_deathsat25"
    ]

    df = df[realtime_fields].copy()
    roles = ["top", "jng", "mid", "bot", "sup"]

    # build blue rows
    blue = game_result[["gameid", "win"] + [f"{r}_blue" for r in roles]].copy()
    blue.rename(columns={f"{r}_blue": r for r in roles}, inplace=True)
    blue["side"] = "Blue"

    # build red rows
    red = game_result[["gameid", "win"] + [f"{r}_red" for r in roles]].copy()
    red.rename(columns={f"{r}_red": r for r in roles}, inplace=True)
    red["side"] = "Red"

    lineup = pd.concat([blue, red], ignore_index=True)

    out = df.merge(lineup, on=["gameid", "side"], how="left")

    out.rename(columns={"win": "match_win"}, inplace=True)
    cols = [
    "goldat25", "xpat25", "csat25",
    "opp_goldat25", "opp_xpat25", "opp_csat25",
    "golddiffat25", "xpdiffat25", "csdiffat25",
    "killsat25", "assistsat25", "deathsat25",
    "opp_killsat25", "opp_assistsat25", "opp_deathsat25"
    ]
    
    mask = out['side'].eq('Red')
    out.loc[mask, 'match_win'] = 1 - out.loc[mask, 'match_win']
    return out[out["datacompleteness"] != "partial"].dropna(subset = cols)


# -----------------------------------------------------------
# save output files
# -----------------------------------------------------------

def save_outputs(game_result, realtime, path="data_clean"):
    '''Save cleaned datasets to disk.'''
    os.makedirs(path, exist_ok=True)

    game_result.to_csv(os.path.join(path, "game_result.csv"), index=False)
    realtime.to_csv(os.path.join(path, "realtime.csv"), index=False)


# -----------------------------------------------------------
# main
# -----------------------------------------------------------

def main():

    print("loading csvs...")
    df_raw = load_raw_csvs()

    print("standardizing columns...")
    df = standardize_columns(df_raw)

    print("splitting rows...")
    players, teams = split_player_and_team_rows(df)

    print("building game_result...")
    game_result = build_result_dataset(players)

    print("building realtime dataset...")
    realtime = build_realtime_dataset(teams, game_result)

    print("saving outputs...")
    save_outputs(game_result, realtime)


if __name__ == "__main__":
    main()
