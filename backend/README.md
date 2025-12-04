# LOL Match Prediction API

This project provides machine-learning–based win predictions for **League of Legends** matches.  
Frontend developers can use this API to display:

- Pick-phase win rates  
- Mid-game win rates (10 / 15 / 20 / 25 minutes)  
- Full-game win rates  
- Feature importance  
- Per-input feature contributions (why the model thinks this way)

---

## 🚀 Start the API

```bash
uvicorn api:app --reload
```

Open interactive docs:

```
http://127.0.0.1:8000/docs
```

---

## 📌 Available Endpoints

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

# 🎮 1. Pick Phase Prediction  
### `POST /predict/lineup`

### Example Request
```json
{
  "bot_blue": "Aphelilios",
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

# 🔥 2. Mid-Game Prediction  
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

# 🏆 3. Full-Game Prediction  
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

---

# 📊 How to Use in Frontend

- Use `p_blue` / `p_blue_norm` as the displayed win rate.
- Use `feature_contribs` to draw bar charts:
  - Positive value → favors blue side
  - Negative value → favors red side
- Show the top 5–10 features for clarity.

---

# 🧪 Test Script

Run all tests:

```bash
python test.py
```

Output will be written to:

```
test_results.txt
```

---

# 📎 Notes

- Pick-phase features are high-dimensional (one-hot). Only show top contributions.
- Mid-game & full-game features are numeric and easy to visualize.
- Probabilities are normalized (blue + red = 1).

---

# ✔ Done
The API is ready for integration with any frontend framework.
