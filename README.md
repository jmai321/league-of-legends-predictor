# League of Legends Pro Play Win Prediction

Predict the outcome of professional League of Legends matches using machine learning.

## Project Video

**Project Video:** [https://youtu.be/ZYgDq5JCPiU](https://youtu.be/ZYgDq5JCPiU)

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

## File Structure
```
league-of-legends-predictor/
├── docker-compose.yml                     # Docker container orchestration
├── README.md
│
├── backend/                               # FastAPI prediction server
│   ├── Dockerfile                         # Backend container configuration
│   ├── README.md
│   ├── api.py                            # Main FastAPI application
│   ├── model.py                          # ML model inference logic
│   ├── requirements.txt                   # Python dependencies
│   ├── graph/                            # Model performance visualizations
│   │   ├── lineup_confusion_matrix.png
│   │   ├── lineup_roc_tuned.png
│   │   ├── realtime_confusion_matrix.png
│   │   ├── realtime_mid10_confusion_matrix.png
│   │   ├── realtime_mid10_roc.png
│   │   ├── realtime_mid15_confusion_matrix.png
│   │   ├── realtime_mid15_roc.png
│   │   ├── realtime_mid20_confusion_matrix.png
│   │   ├── realtime_mid20_roc.png
│   │   ├── realtime_mid25_confusion_matrix.png
│   │   ├── realtime_mid25_roc.png
│   │   └── realtime_roc.png
│   └── model/                            # Trained XGBoost models
│       ├── lineup_model.joblib           # Draft prediction model
│       ├── realtime_model.joblib         # Live prediction model (full game)
│       ├── realtime_mid10_model.joblib   # 10-minute prediction model
│       ├── realtime_mid15_model.joblib   # 15-minute prediction model
│       ├── realtime_mid20_model.joblib   # 20-minute prediction model
│       └── realtime_mid25_model.joblib   # 25-minute prediction model
│
├── frontend/                              # Next.js web application
│   ├── Dockerfile                         # Frontend container configuration
│   ├── README.md
│   ├── package.json                       # Node.js dependencies
│   ├── package-lock.json
│   ├── next.config.ts                     # Next.js configuration
│   ├── tsconfig.json                      # TypeScript configuration
│   ├── eslint.config.mjs                  # ESLint configuration
│   ├── postcss.config.mjs                 # PostCSS configuration
│   ├── components.json                    # shadcn/ui component configuration
│   ├── next-env.d.ts                      # Next.js type definitions
│   ├── app/                              # Next.js app router pages
│   │   ├── draft/
│   │   │   └── page.tsx                  # Draft prediction interface
│   │   ├── live/
│   │   │   └── page.tsx                  # Live prediction interface
│   │   ├── layout.tsx                    # Root layout component
│   │   ├── page.tsx                      # Home page
│   │   ├── globals.css                   # Global styles
│   │   └── favicon.ico
│   ├── components/                       # React components
│   │   ├── ui/                          # shadcn/ui components
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── checkbox.tsx
│   │   │   ├── input.tsx
│   │   │   ├── label.tsx
│   │   │   ├── select.tsx
│   │   │   └── tabs.tsx
│   │   ├── layout/
│   │   │   └── PageLayout.tsx            # Layout wrapper component
│   │   ├── ChampionSelector.tsx          # Champion selection dropdown
│   │   ├── ModelSelector.tsx             # Model selection component
│   │   ├── Navigation.tsx                # Navigation component
│   │   └── TeamStatsForm.tsx             # Live game statistics input
│   ├── api/                              # Frontend API client
│   │   ├── client.ts
│   │   ├── draft.ts
│   │   ├── index.ts
│   │   └── live.ts
│   ├── constants/                        # Game data constants
│   │   ├── champions.ts
│   │   └── gameStats.ts
│   ├── lib/                              # Utility functions
│   │   ├── transformers.ts
│   │   └── utils.ts
│   └── types/                            # TypeScript type definitions
│       └── index.ts
│
├── pipeline/                              # Data cleaning and preprocessing
│   ├── README.md
│   └── src/
│       └── data_cleaning.py              # Data processing script
│
└── notebooks/                            # Jupyter analysis notebooks
    ├── Readme.md
    ├── requirement.txt                   # Jupyter dependencies
    └── visualization.ipynb               # Data exploration and model analysis
```

## Third-Party Modules Used

**Backend:**
- FastAPI - Web framework for building APIs
- XGBoost - Gradient boosting machine learning library
- scikit-learn - Machine learning utilities and metrics
- pandas - Data manipulation and analysis
- uvicorn - ASGI web server

**Frontend:**
- Next.js 16.0.7 - React framework
- React 19.2.0 - UI library
- TypeScript - Type-safe JavaScript
- Tailwind CSS 4 - Utility-first CSS framework
- Radix UI - Headless UI components (@radix-ui/react-*)
- Lucide React - Icon library
- class-variance-authority - Component variant utilities
- clsx - Conditional className utility
- tailwind-merge - Tailwind CSS class merging

**Data Pipeline:**
- pandas - Data processing
- numpy - Numerical computations

**Notebooks:**
- jupyter - Interactive notebook environment
- matplotlib - Data visualization
- seaborn - Statistical data visualization

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
- Aaron(Xuhang)
- Gary
- Xiaojun
- Tingyun
