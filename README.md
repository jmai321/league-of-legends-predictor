# League of Legends Pro Play Win Prediction

Predict the outcome of professional League of Legends matches using machine learning.

## Setup

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Backend
```bash
cd backend
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn api:app --reload
```
*Note: Requires Python 3.10. Other versions may have compatibility issues.*

## Project Structure
```
lol-win-prediction/
├── pipeline/             # Data cleaning + model training
├── backend/              # FastAPI prediction server
├── frontend/             # Next.js web application
├── notebooks/            # Visualizations and analysis
├── README.md
```

## Data Source
Raw CSVs available [here](https://drive.google.com/drive/u/0/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH).

## Team
- Jonathan
- Aaron
- Gary
- Xiaojun
- Tingyun
