# Civilization Online RPG

A player-hosted online grand strategy RPG game inspired by **Romance of the Three Kingdoms 14** and **Europa Universalis 4**.

## Features

### Game Mechanics (RTK14-Inspired)
- **Province-based Map**: Hex-grid map with terrain types (plains, mountains, forest, desert, river, coast, marsh)
- **Character/Officer System**: Officers with 5 core stats (Command, Force, Intelligence, Politics, Charisma)
- **Character Roles**: Ruler, General, Strategist, Governor, Diplomat, Spy
- **Leveling & Experience**: Officers gain XP from battles and actions
- **Loyalty & Defection**: Officers may defect if loyalty drops too low

### Game Mechanics (EU4-Inspired)
- **Faction Management**: Gold, food, manpower resources; tax rates, conscription
- **Technology Tree**: 4 branches (Military, Economy, Culture, Diplomacy) with 27 technologies
- **Diplomacy System**: Alliance, War, Trade, Truce, Vassal relationships
- **Province Development**: Buildings (Farm, Market, Barracks, Wall, Academy, Temple, Workshop, Port)
- **War & Peace**: Army movement, siege mechanics, terrain combat bonuses
- **Stability & Legitimacy**: Government stability, war exhaustion
- **Random Events**: 11 event types (plagues, harvests, rebellions, festivals, etc.)
- **Seasonal Effects**: Spring/Summer/Autumn/Winter affect food production and recruitment

### Multiplayer
- **WebSocket Real-time**: Live multiplayer via WebSocket connections
- **Player-hosted Servers**: Players can host their own game servers
- **Turn-based with Ready System**: Players submit actions, then end turn
- **In-game Chat**: Real-time chat between players
- **Game Rooms**: Multiple game rooms on a single server
- **Save/Load**: Persistent game state via JSON save files

### Technical
- **Zero External Database**: Uses JSON for persistence (no database setup required)
- **Browser-based Client**: No client installation needed - just open a browser
- **Lightweight Server**: Runs on any machine with Python 3.8+
- **Docker Support**: One-command deployment with Docker
- **Free Hosting Compatible**: Works on Replit, Railway, Render, Fly.io, etc.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-server.txt
```

### 2. Start the Server

```bash
python run_server.py
```

The server starts on `http://localhost:8000`. Open this URL in your browser to play.

### 3. LAN Play

To allow other players on your local network:

```bash
python run_server.py --host 0.0.0.0 --port 8000
```

Share your local IP address with other players (e.g., `http://192.168.1.100:8000`).

### 4. Docker Deployment

```bash
docker build -t civ-rpg .
docker run -p 8000:8000 civ-rpg
```

## Free Hosting Options

Players can host servers for free on these platforms:

| Platform | Free Tier | Setup |
|----------|-----------|-------|
| [Replit](https://replit.com) | Always free | Fork and run |
| [Railway](https://railway.app) | $5 credit/month | Connect GitHub repo |
| [Render](https://render.com) | Free web services | Connect GitHub repo |
| [Fly.io](https://fly.io) | 3 free VMs | `fly launch` |
| [Oracle Cloud](https://cloud.oracle.com) | Always Free VMs | Deploy Docker |
| [Google Cloud](https://cloud.google.com) | Free tier | Cloud Run |

### Replit Deployment

1. Fork this repository on Replit
2. Set the run command to: `python run_server.py --port 3000`
3. Click Run

### Railway Deployment

1. Connect your GitHub repository
2. Railway auto-detects the Dockerfile
3. Deploy automatically

## API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | Server status |
| GET | `/api/rooms` | List game rooms |
| POST | `/api/rooms/create` | Create a new room |
| POST | `/api/rooms/{id}/start` | Start a game |
| GET | `/api/rooms/{id}/state` | Get game state |
| GET | `/api/rooms/{id}/tech_tree` | Get technology tree |
| POST | `/api/rooms/{id}/save` | Save game |
| GET | `/api/saves` | List save files |

### WebSocket Protocol

Connect to `ws://server:port/ws/{room_id}?player_name=NAME&player_id=ID`

#### Client Messages

```json
{"type": "action", "action": {"type": "build", "province_id": "...", "building_type": "farm"}}
{"type": "action", "action": {"type": "move_army", "army_id": "...", "target_province_id": "..."}}
{"type": "action", "action": {"type": "recruit_army", "province_id": "...", "soldiers": 1000}}
{"type": "action", "action": {"type": "diplomacy", "target_faction_id": "...", "action": "war"}}
{"type": "action", "action": {"type": "research", "tech_id": "mil_1"}}
{"type": "ready"}
{"type": "chat", "text": "Hello!"}
{"type": "get_state"}
{"type": "force_turn"}
```

#### Server Messages

```json
{"type": "connected", "player_id": "...", "state": {...}}
{"type": "game_started", "state": {...}}
{"type": "turn_processed", "turn": 1, "events": [...], "state": {...}}
{"type": "chat", "name": "Player", "text": "Hello!"}
{"type": "player_joined", "name": "Player"}
{"type": "player_left", "name": "Player"}
```

## Architecture

```
online_rpg/
  __init__.py          # Package init
  models.py            # Data models (Province, Character, Faction, Army, etc.)
  map_generator.py     # Hex-grid map generation with terrain
  tech_tree.py         # Technology tree (27 techs, 4 branches)
  game_engine.py       # Core game logic (turns, combat, economy, diplomacy)
  server.py            # FastAPI + WebSocket server
  static/
    index.html         # Game client HTML
    style.css          # Game client styles
    game.js            # Game client JavaScript (map rendering, UI, WebSocket)
```

## Game Controls

- **Click** on a province to select it and view details
- **Scroll** to zoom the map
- **Drag** to pan the map
- Use the **side panel tabs** to switch between Province, Army, Officers, Diplomacy, and Tech views
- Click **End Turn** to advance the game

## License

MIT License
