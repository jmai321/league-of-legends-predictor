# Data Cleaning

## Setup
Create a folder named data_clean and data_raw
Download the csv files from OraclesElixir database https://drive.google.com/drive/u/1/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH
Store the csv files into dara_raw folder

1. Inside the `src/` directory, create the following folders:
    data_raw/  (store downloaded Oracle’s Elixir CSV files here)
    data_clean/ (cleaned CSV files will be written here) 
2. Download the Oracle’s Elixir dataset from:
https://drive.google.com/drive/u/1/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH
3. Move all downloaded `.csv` files into the `data_raw/` folder.

## How to Run
python pipeline/src/data_cleaning.py

## File Structure
pipeline/
  README.md
  src/
    data_cleaning.py
    data_raw/
      raw CSVs from Oracle’s Elixir
    data_clean/
      game_result.csv
      realtime.csv