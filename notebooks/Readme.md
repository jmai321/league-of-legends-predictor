# Notebooks - Data Analysis and Visualization

Jupyter notebook for exploring League of Legends match data and analyzing model performance.

## How to Run

1. **Complete data cleaning first:**
   ```bash
   # See pipeline/README.md for instructions
   python pipeline/src/data_cleaning.py
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirement.txt
   pip install jupyter
   ```

3. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

4. **Open and run `visualization.ipynb`**

## File Structure

```
notebooks/
├── Readme.md
├── requirement.txt           # Analysis dependencies
└── visualization.ipynb       # Main data analysis notebook
```

## Notebook Contents

**visualization.ipynb** contains:
- **Data Exploration:** Champion pick rates, win rates, and team composition analysis
- **Performance Metrics:** Model accuracy, confusion matrices, and ROC curves
- **Feature Analysis:** Most important features for predictions across different time points
- **Game Dynamics:** How objectives, gold differences, and other metrics impact win probability
- **Time-Series Analysis:** How prediction accuracy improves from draft to late game

## Analysis Topics

- Champion selection patterns and effectiveness
- Team synergy effects (bot lane combinations)
- Side advantages (Blue vs Red team performance)
- Objective importance (Baron, Dragon, Herald impact)
- Economic advantages (Gold and XP difference effects)
- Comeback probability analysis
- Tower control and map progression

## Third-Party Dependencies

- **matplotlib** - Data visualization and plotting
- **seaborn** - Statistical data visualization  
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **jupyter** - Interactive notebook environment
