<div align="center">
  <h1>FCML Studio: Machine Learning Fantasy Cricket Optimizer</h1>
  <p>A Data-Driven Lineup Generator Built on Operations Research, Contextual AI, and Advanced Analytics Tailored for Daily Fantasy Sports.</p>

  [![React](https://img.shields.io/badge/React-18.x-61DAFB?logo=react&logoColor=black)](https://reactjs.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org/)
  [![TailwindCSS](https://img.shields.io/badge/Tailwind-3.0-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
</div>

<hr/>

## About The Project

Generating high-yield Daily Fantasy lineups (such as for Dream11) is mathematically defined as a 0-1 Knapsack Optimization problem. **FCML Studio** transforms raw, ball-by-ball performance data into structured, rule-compliant combinatorial lineages through deterministic machine learning and integer linear programming.

This engine separates itself from conventional statistical analyzers by engineering **Contextual Matrix Optimization**. It manipulates base projected fantasy thresholds in real-time according to live stadium environments—autonomously restructuring the probability weight of secondary spinners, for example, when heavy dew factors restrict their efficacy.

## Architecture & Features

### 1. Operations Research (PuLP Integer Linear Programming)
The engine executes multi-variable linear constraints to guarantee strictly valid arrays:
- **Financial Bound:** Aggregated player credits strictly `≤ 100`.
- **Structural Bounds:** Hard limits on standard fantasy roster configuration (e.g., 1-4 Wicket Keepers, 3-6 Batters, 1-4 All-Rounders, 3-6 Bowlers) while ensuring a strict `11` player total.
- **Franchise Bounds:** Restricts player domination to a maximum of `7` selections from any single franchise entity to retain lineup variance.

### 2. Contextual Environment Modifiers
Live Dropdowns allow the user to inject real-world context data into the model prior to execution. By parsing factors such as *Spin Pitch*, *Pace Pitch*, *Overcast Weather*, and *Heavy Dew*, the algorithm scales player point projections upward or downward mathematically, actively predicting how specific roles will fare before the toss.

### 3. Automated Data Ingestion Workflow
Instead of static historical datasets, the system executes an automated Python-based scraping routine to circumvent proxy protections, fetch live cricket fixture matrices, process stadium mapping relationships, and write actionable telemetry directly to the backend database endpoints.

### 4. Bulk CSV Export Node
Constructed for professional-level Daily Fantasy participants, the system generates continuous sequential matrices (up to 20 unique line-ups per cycle) and provides instantaneous `.csv` compilation for zero-friction bulk uploading directly to provider databases.

### 5. Frontend Telemetry Dashboard
A proprietary, interactive user interface heavily optimized utilizing React and Tailwind. The layout includes:
- **Lock/Ban Matrix:** Explicitly force must-have players into the integer logic equation or permanently eliminate them.
- **Deep Analytics HUD:** Dynamically generated Rechart components plot recent execution trajectory, economy averages, and micro-matchup trajectories cleanly onto the screen.

## Technology Stack

**Frontend Framework**
* **React.js** (Declarative interactive UI)
* **Vite** (Next-generation frontend tooling and fast-bundling)
* **Tailwind CSS** (Utility-first framework architecture)
* **Recharts** (Declarative data visualization charting)

**Backend Intelligence**
* **FastAPI** (High-performance async server frameworks)
* **Pandas** (Analytical data manipulation)
* **PuLP** (Linear programming model execution)
* **CloudScraper & Regex** (Automated DOM parsing and bot-protection circumvention)

<br/>

## Local Deployment Instructions

Follow these commands to deploy the environment locally across two dependent shells.

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