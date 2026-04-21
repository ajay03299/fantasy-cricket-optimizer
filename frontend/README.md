# IPL 2026 Fantasy Cricket Optimizer Frontend

This is the React frontend for the IPL 2026 Fantasy Cricket Optimizer, built with Vite, React, and Tailwind CSS.

## Features

- Interactive dashboard for fantasy cricket lineup optimization
- Real-time player filtering based on IPL 2026 active players
- Lock/Ban functionality for player constraints
- Environmental modifiers (pitch type, weather conditions)
- Bulk lineup generation and CSV export
- Analytics dashboard with player statistics and matchups

## Getting Started

### Prerequisites
- Node.js 18+
- npm or yarn

### Installation
```bash
npm install
```

### Development
```bash
npm run dev
```

### Build for Production
```bash
npm run build
```

## Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Styling
- **Recharts** - Data visualization
- **React Router** - Navigation

## Project Structure

```
frontend/
├── public/          # Static assets
├── src/
│   ├── components/  # Reusable UI components
│   ├── pages/       # Page components (Dashboard, Matches, etc.)
│   ├── utils/       # Utility functions
│   └── assets/      # Images and icons
├── package.json
└── vite.config.js
```

## API Integration

The frontend communicates with the FastAPI backend running on `http://localhost:8000`. Key endpoints:

- `GET /matches` - Get available matches
- `GET /players/{match_id}` - Get players for a match
- `POST /optimize` - Run lineup optimization
- `GET /player_stats/{match_id}/{player_name}` - Get detailed player stats
