<div align="center">
  <h1>FCML Studio: IPL 2026 Fantasy Cricket Optimizer</h1>
  <p>A Data-Driven Lineup Generator for IPL 2026 Built on Operations Research, Contextual AI, and Advanced Analytics for Daily Fantasy Sports.</p>

  [![React](https://img.shields.io/badge/React-18.x-61DAFB?logo=react&logoColor=black)](https://reactjs.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org/)
  [![TailwindCSS](https://img.shields.io/badge/Tailwind-3.0-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
  [![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF6B35?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
</div>

<hr/>

## About The Project

Generating high-yield Daily Fantasy lineups for IPL 2026 is mathematically defined as a 0-1 Knapsack Optimization problem. **FCML Studio** transforms IPL 2026 ball-by-ball performance data into structured, rule-compliant combinatorial lineups through deterministic machine learning and integer linear programming.

This engine separates itself from conventional statistical analyzers by engineering **Contextual Matrix Optimization** with real-time IPL 2026 data. It manipulates base projected fantasy thresholds according to live stadium environments and player performance in the current season.

## Key Features

### 1. IPL 2026 Active Player Filtering
- Automatically filters players to only those who have appeared in IPL 2026 matches
- Uses expected playing XIs from recent matches for upcoming fixtures
- Ensures lineup optimization focuses on currently active players

### 2. Operations Research (PuLP Integer Linear Programming)
The engine executes multi-variable linear constraints to guarantee strictly valid arrays:
- **Financial Bound:** Aggregated player credits strictly `≤ 100`.
- **Structural Bounds:** Hard limits on standard fantasy roster configuration (e.g., 1-4 Wicket Keepers, 3-6 Batters, 1-4 All-Rounders, 3-6 Bowlers) while ensuring a strict `11` player total.
- **Franchise Bounds:** Restricts player domination to a maximum of `7` selections from any single franchise entity to retain lineup variance.

### 3. Contextual Environment Modifiers
Live Dropdowns allow the user to inject real-world context data into the model prior to execution. By parsing factors such as *Spin Pitch*, *Pace Pitch*, *Overcast Weather*, and *Heavy Dew*, the algorithm scales player point projections upward or downward mathematically, actively predicting how specific roles will fare before the toss.

### 4. Advanced ML Model (XGBoost)
- Trained on comprehensive IPL historical data
- Incorporates IPL 2026 season performance for improved predictions
- Features engineered from ball-by-ball data including recent form, venue statistics, and matchup analysis

### 5. Automated Data Pipeline
- Processes IPL 2026 deliveries data for active player identification
- Integrates team squad information with performance data
- Generates predictions using latest model versions

### 6. Bulk CSV Export
Constructed for professional-level Daily Fantasy participants, the system generates continuous sequential matrices (up to 20 unique line-ups per cycle) and provides instantaneous `.csv` compilation for zero-friction bulk uploading directly to provider databases.

### 7. Interactive Frontend Dashboard
A proprietary, interactive user interface heavily optimized utilizing React and Tailwind. The layout includes:
- **Lock/Ban Matrix:** Explicitly force must-have players into the integer logic equation or permanently eliminate them.
- **Deep Analytics HUD:** Dynamically generated Rechart components plot recent execution trajectory, economy averages, and micro-matchup trajectories cleanly onto the screen.
- **Real-time Player Filtering:** Only displays players expected to play in upcoming matches.

## Technology Stack

**Frontend Framework**
* **React.js** (Declarative interactive UI)
* **Vite** (Next-generation frontend tooling and fast-bundling)
* **Tailwind CSS** (Utility-first framework architecture)
* **Recharts** (Declarative data visualization charting)

**Backend Intelligence**
* **FastAPI** (High-performance async server frameworks)
* **Pandas** (Analytical data manipulation)
* **XGBoost** (Gradient boosting machine learning)
* **PuLP** (Linear programming model execution)
* **Scikit-learn** (Machine learning utilities)

**Data Sources**
* IPL 2026 Ball-by-ball deliveries data
* Team squad information
* Historical IPL performance data
* Venue and environmental statistics

<br/>

<br/>

## Data Sources

The system utilizes several key data sources for IPL 2026 optimization:

- **IPL 2026 Deliveries**: Ball-by-ball data from current season matches (`IPL_2026/ipl_2026_deliveries.csv`)
- **Team Squads**: Official team player lists (`Team_Squad/active_players_squad_full_names.csv`)
- **Historical Data**: Past IPL seasons for model training (`Cleaned_Dataset/`, `Raw_Dataset/`)
- **Match Schedule**: Upcoming fixtures (`data/upcoming_matches.csv`)
- **Model Predictions**: Latest XGBoost predictions (`output/best_model_v*_predictions.csv`)

## Model Training

The XGBoost model is trained using historical IPL data with engineered features:

```bash
cd src
python train_xgb_v6.py
```

Key features include:
- Recent form (last 5 matches)
- Venue-specific performance
- Head-to-head statistics
- Player roles and consistency metrics

## Data Preparation

### Active Player Filtering
The system automatically identifies active players from IPL 2026 deliveries data:

```bash
# Extract active players from deliveries
python -c "
import pandas as pd
df = pd.read_csv('IPL_2026/ipl_2026_deliveries.csv')
active_players = set(df['striker'].dropna()) | set(df['bowler'].dropna())
print('Active players:', len(active_players))
"
```

### Feature Engineering
Run feature table generation for model input:

```bash
python build_feature_table_v6.py
```

<br/>

## Local Deployment Instructions

Follow these commands to deploy the environment locally across two dependent shells.

### Prerequisites
- Python 3.10+
- Node.js 18+
- pip and npm installed

### 1. Backend Server Initialization
Navigate to the `/api` directory to install dependencies and boot the server application.
```bash
cd api

# Install backend dependencies
pip install -r requirements.txt

# Execute the ASGI Server locally
python main.py
```
*The host server will bind to `http://127.0.0.1:8000`.*

### 2. Frontend Render Initialization
Open an adjacent shell and navigate to the `/frontend` directory.
```bash
cd frontend

# Install exact node modules
npm install

# Initialize Vite server environment
npm run dev
```
*The React UI render is actively hosted and accessible at `http://localhost:5173`.*

## Active Usage Workflow
1. Navigate via the dashboard to verify the active parsed schedule match.
2. Formulate your mathematical constraints via the `Weather` and `Pitch` modifiers in the Optimization controls.
3. Utilize the **Lock** or **Ban** nodes explicitly to limit the problem-space of the engine based on intrinsic player-knowledge.
4. Activate the **Run ML Optimizer** engine to compile linear solutions.
5. Set to render multi-lineage output (e.g., 20 Lineups) and execute **Export CSV** to cleanly package your deployment structure.

## Project Structure

```
Fantasy-Cricket-Optimizer-main/
├── api/                    # FastAPI backend
│   ├── main.py            # Main API server
│   ├── ipl2026_loader.py  # IPL 2026 data processing
│   └── requirements.txt
├── frontend/              # React frontend
│   ├── src/
│   └── package.json
├── src/                   # Model training scripts
│   ├── train_xgb_v6.py
│   └── build_feature_table_v6.py
├── IPL_2026/              # Current season data
├── Team_Squad/            # Team roster data
├── data/                  # Processed datasets
├── output/                # Model predictions
└── Cleaned_Dataset/       # Historical data
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- IPL data sources
- XGBoost for machine learning
- FastAPI and React communities
- Fantasy cricket community for optimization techniques