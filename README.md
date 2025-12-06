# League of Legends Pro Play Win Prediction

Predict the outcome of professional League of Legends matches using machine learning.

## Setup

### Docker

**Requirements:** Docker Desktop (if you do not want to use docker you can run the frontend/backend individually as explained below)

The easiest way to run the entire application:

```bash
git clone [repository-url]
cd league-of-legends-predictor
docker-compose up --build
```

Then visit http://localhost:3000 to use the application.

**To stop:** `docker-compose down`

### Manual Setup

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

#### Backend
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

## Detailed Setup Instructions

For component-specific setup and usage details:

- **[Frontend README](frontend/README.md)** - Next.js web application setup and development
- **[Backend README](backend/README.md)** - FastAPI server configuration and API details
- **[Pipeline README](pipeline/README.md)** - Data cleaning and model training pipeline
- **[Notebooks README](notebooks/README.md)** - Jupyter analysis and visualization setup

## Data Source
Raw CSVs available [here](https://drive.google.com/drive/u/0/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH).

## Team
- Jonathan
- Aaron
- Gary
- Xiaojun
- Tingyun
