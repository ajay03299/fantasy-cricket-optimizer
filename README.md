<div align="center">
  <img src="https://media.istockphoto.com/id/1141191007/vector/sports-concept-with-cricket-equipment.jpg?s=612x612&w=0&k=20&c=B3a6p-h_6O7qYt9V4C7d1l7B6o7B3c6p-h_6O7B3c6=" width="200" alt="FCML Studio Logo"/>
  <h1>🏏 FCML Studio: Machine Learning Fantasy Cricket Optimizer</h1>
  <p>A Professional-Grade Lineup Generator built on Operations Research, Contextual AI, and Advanced Analytics tailored for Daily Fantasy Sports.</p>

  [![React](https://img.shields.io/badge/React-18.x-61DAFB?logo=react&logoColor=black)](https://reactjs.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org/)
  [![TailwindCSS](https://img.shields.io/badge/Tailwind-3.0-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
</div>

<hr/>

## 📖 About The Project

Generating winning Daily Fantasy lineups (like Dream11) is no longer about guessing; it's a mathematical optimization problem. **FCML Studio** transforms raw, ball-by-ball cricket data into mathematically flawless structural lineages using the **`0-1 Knapsack` Algorithm**.

Unlike basic statistical apps, this suite natively features **Contextual AI**. It alters projected fantasy points in real-time based on live stadium conditions, such as reducing the scoring probability of 2nd-innings spinners if **"Heavy Dew"** is selected during a night match. 

## ✨ Key Features

* 🧠 **Mathematical PuLP Solver (Knapsack):** Binds to real-world structural constraints (Strict 100 Credit Cap, exact positional arrays of 1-4 WKs / 3-6 BATs / 1-4 ARs / 3-6 BOWLs, capping at max 7 players per franchise).
* ⛅ **Contextual Environmental Modifiers:** Live Dropdowns for *Spin Pitches*, *Pace Pitches*, *Overcast Weather*, and *Heavy Dew* dynamically mutate projected player scores instantaneously before optimization. 
* 🕷️ **Automated Schedule Web Scraper:** Uses automated Python-regex routines to fetch and synchronize live current season data, mapping completed and upcoming matches straight to the dashboard.
* ⚡ **Bulk CSV Exporter:** Supports competitive DFS players by sequentially wrapping and exporting up to 20 optimized line-ups to a localized `.csv` file for instant bulk-uploading.
* 📊 **Deep Analytics HUD:** React Recharts generates *Powerplay / Middle Over / Death Over* execution Radar charts and Recent Form sparklines to graphically demystify the Optimizer's behavior.

## 🛠️ Technology Stack

**Frontend Framework**
* **React.js** (Hook-driven Interactive UI)
* **Vite** (Lightning-fast HMR and Bundling)
* **Tailwind CSS** (Utility-first styling, neon-cyberpunk aesthetic)
* **Recharts** & **Lucide React** (Dynamic Visualizations & SVG icon handling)

**Backend Intelligence**
* **FastAPI** (Instantaneous JSON delivery and async route optimization)
* **Pandas** (High-intensity matrix and Dataframe engineering)
* **PuLP** (Operations Research Library executing the Integer Linear Programming logic)
* **CloudScraper & Regex** (Circumvents 403 blocks for persistent schedule fetching)

<br/>

## 🚀 Getting Started

Follow these steps to run the Optimizer locally on your machine. 

### Prerequisites
* [Node.js](https://nodejs.org/en/) (v16.0+)
* [Python](https://www.python.org/downloads/) (v3.10+)

### 1. Backend Setup (FastAPI & ML)
Open your terminal and navigate to the `/api` directory.
```bash
cd api

# Install required mathematical and API dependencies
pip install -r requirements.txt

# Boot up the Uvicorn Local Server
python main.py
```
*The server will spin up on `http://127.0.0.1:8000`.*

### 2. Frontend Setup (React UI)
Open a new terminal session and navigate to the `/frontend` directory.
```bash
cd frontend

# Install the Node packages
npm install

# Start the Vite Hot-Reloading server
npm run dev
```
*Your interactive dashboard is now cleanly hosted on `http://localhost:5173`!*

<br/>

## 🧩 Usage Guide

1. **Dashboard Initialization:** Once loaded, navigate to the **Matches** pane to observe the scraped active Schedule. 
2. **Setup your Environment:** 
   * Hover over the `Weather` and `Pitch` toggles on the Optimizer screen to inject real-world context (If Heavy Dew is selected, be sure to utilize the **Toss** dropdown to lock in the team batting first!).
3. **Impose your Logic:** 
   * Click the 🔒 icon to Force a must-have player into the lineup, or the 🚫 icon to completely fade/ban a player whose form you distrust.
4. **Generate Matrix:** Hit *Run ML Optimizer* to solve your structural arrays.
5. **Download and Deploy:** Adjust the lineage count to 20, and click **Export CSV** to prep your download directly for any DFS hosting platform!

## 🤝 Contribution
If you have suggestions on bringing Deep Neural Net predictions or Live Endpoints (WebSockets) to the engine, feel free to Fork the project and open a Pull Request!