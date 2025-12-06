# Backend - League of Legends Match Prediction API

FastAPI backend providing machine learning-based win predictions for League of Legends matches using XGBoost models.

## How to Run

**Requirements:** Python 3.10

```bash
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn api:app --reload
```

The API will be available at: http://127.0.0.1:8000

Interactive API documentation: http://127.0.0.1:8000/docs

## File Structure

```
backend/
├── Dockerfile                      # Docker container configuration  
├── README.md
├── requirements.txt                 # Python dependencies
├── api.py                          # Main FastAPI application
├── model.py                        # ML model inference logic
│
├── graph/                          # Model performance visualizations
│   ├── lineup_confusion_matrix.png
│   ├── lineup_roc_tuned.png
│   ├── realtime_confusion_matrix.png
│   ├── realtime_mid10_confusion_matrix.png
│   ├── realtime_mid10_roc.png
│   ├── realtime_mid15_confusion_matrix.png
│   ├── realtime_mid15_roc.png
│   ├── realtime_mid20_confusion_matrix.png
│   ├── realtime_mid20_roc.png
│   ├── realtime_mid25_confusion_matrix.png
│   ├── realtime_mid25_roc.png
│   └── realtime_roc.png
│
└── model/                          # Trained XGBoost models
    ├── lineup_model.joblib         # Draft prediction model
    ├── realtime_model.joblib       # Full game prediction model
    ├── realtime_mid10_model.joblib # 10-minute prediction model
    ├── realtime_mid15_model.joblib # 15-minute prediction model
    ├── realtime_mid20_model.joblib # 20-minute prediction model
    └── realtime_mid25_model.joblib # 25-minute prediction model
```

## Third-Party Dependencies

- **FastAPI 0.123.9** - Modern web framework for building APIs
- **Uvicorn 0.38.0** - ASGI web server for running FastAPI
- **Pydantic 2.12.5** - Data validation using Python type annotations
- **XGBoost 3.1.2** - Gradient boosting machine learning library
- **scikit-learn 1.2.1** - Machine learning utilities and metrics
- **pandas 2.3.3** - Data manipulation and analysis
- **numpy 1.24.4** - Numerical computing
- **joblib 1.5.2** - Model serialization and loading
- **matplotlib 3.10.7** - Plotting library (for model visualization)

---

## Available Endpoints

| Endpoint | Description |
|----------|-------------|
| **POST /predict/lineup** | Predict win rate from champion picks |
| **POST /predict/realtime/full** | Predict win rate from full-game stats |
| **POST /predict/realtime/mid/10** | 10-minute mid-game prediction |
| **POST /predict/realtime/mid/15** | 15-minute mid-game prediction |
| **POST /predict/realtime/mid/20** | 20-minute mid-game prediction |
| **POST /predict/realtime/mid/25** | 25-minute mid-game prediction |

All endpoints return:
- Blue side win rate
- Red side win rate
- Global feature importance
- Per-input feature contributions

---

# 1. Pick Phase Prediction  
### `POST /predict/lineup`

### Example Request
```json
{
  "bot_blue": "Aphelios",
  "jng_blue": "LeeSin",
  "mid_blue": "Ahri",
  "sup_blue": "Thresh",
  "top_blue": "Aatrox",
  "bot_red": "Xayah",
  "jng_red": "Viego",
  "mid_red": "Syndra",
  "sup_red": "Rakan",
  "top_red": "Renekton"
}
```

### Example Response
```json
{
  "p_blue": 0.54,
  "p_red": 0.45,
  "top_features": [...],
  "feature_contribs": [...]
}
```

---

# 2. Mid-Game Prediction  
### `POST /predict/realtime/mid/{minute}`  
Supports: **10, 15, 20, 25**

### Example Request (10 min)
```json
{
  "rows": [
            {
                "gameid": "TEST_GAME_MID10_1",
                "side": "Blue",
                "teamname": "BlueTeam",
                "goldat10": 12000,
                "xpat10": 8000,
                "csat10": 400,
                "golddiffat10": 800,
                "xpdiffat10": 500,
                "csdiffat10": 20,
                "killsat10": 4,
                "deathsat10": 1,
            },
            {
                "gameid": "TEST_GAME_MID10_1",
                "side": "Red",
                "teamname": "RedTeam",
                "goldat10": 11200,
                "xpat10": 7500,
                "csat10": 380,
                "golddiffat10": -800,
                "xpdiffat10": -500,
                "csdiffat10": -20,
                "killsat10": 1,
                "deathsat10": 4,
            },
        ]
}
```

### Example Response
```json
{
  "minute": 10,
  "p_blue_norm": 0.80,
  "p_red_norm": 0.20,
  "top_features": [...],
  "feature_contribs_blue": [...]
}
```

---

# 3. Full-Game Prediction  
### `POST /predict/realtime/full`

### Example Request
```json
{
  "rows":  [
            {
                "gameid": "GAME_FULL_1",
                "side": "Blue",
                "teamname": "TeamA",
                "gold": 60000,
                "kills": 15,
                "deaths": 8,
                "towers": 8,
            },
            {
                "gameid": "GAME_FULL_1",
                "side": "Red",
                "teamname": "TeamB",
                "gold": 55000,
                "kills": 10,
                "deaths": 15,
                "towers": 3,
            },
        ]
}
```

### Example Response
```json
{
  "p_blue_norm": 0.66,
  "p_red_norm": 0.33,
  "top_features": [...],
  "feature_contribs_blue": [...],
  "feature_contribs_red": [...]
}
```
