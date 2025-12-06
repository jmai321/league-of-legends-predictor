# Pipeline - Data Cleaning and Preprocessing

Data cleaning pipeline for processing League of Legends esports match data from Oracle's Elixir.

## How to Run

1. **Download Data:**
   - Download CSV files from [Oracle's Elixir database](https://drive.google.com/drive/u/1/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH)
   - Create `data_raw/` directory in the pipeline folder
   - Place all downloaded CSV files in `data_raw/`

2. **Create Output Directory:**
   ```bash
   mkdir data_clean
   ```

3. **Run Data Cleaning:**
   ```bash
   python src/data_cleaning.py
   ```

## File Structure

```
pipeline/
├── README.md
├── src/
│   └── data_cleaning.py          # Main data processing script
├── data_raw/                     # Raw CSV files (create this directory)
│   └── [Oracle's Elixir CSV files]
└── data_clean/                   # Output cleaned data (created by script)
    ├── game_result.csv          # Final match outcomes and champion picks
    └── realtime.csv             # Timestamped performance data
```

## Output Files

**game_result.csv:**
- Final game outcomes and champion selections
- Used for draft prediction model training
- Contains team compositions and match results

**realtime.csv:**
- Real-time snapshots at 10, 15, 20, 25-minute intervals  
- Used for live prediction model training
- Contains gold, experience, CS, and objective data over time

## Data Processing Steps

1. **Standardize column names** across different CSV formats
2. **Separate player-level and team-level statistics**
3. **Remove incomplete records** missing timestamp data
4. **Aggregate team performance metrics** by game and time interval
5. **Export cleaned datasets** for model training

## Third-Party Dependencies

- **pandas** - Data manipulation and CSV processing
- **numpy** - Numerical operations and data transformation

*Note: Install dependencies with:*
```bash
pip install pandas numpy
```