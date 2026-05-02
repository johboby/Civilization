"""Game server - FastAPI + WebSocket multiplayer server.

Players can host this server on their own machines or free cloud services.
Supports real-time multiplayer via WebSocket connections.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
except ImportError:
    raise ImportError(
        "FastAPI is required. Install with: pip install fastapi uvicorn[standard]"
    )

from .game_engine import GameEngine, GameState
from .models import DiplomacyStatus
from .tech_tree import get_available_techs, build_tech_tree
from .character_portraits import generate_portrait_svg
from .advanced_systems import list_scenarios, get_scenario

logger = logging.getLogger("rpg_server")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Server configuration
# ---------------------------------------------------------------------------

SAVE_DIR = Path("saves")
STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------


@dataclass
class PlayerConnection:
    """Represents a connected player."""
    player_id: str
    faction_id: Optional[str] = None
    websocket: Optional[WebSocket] = None
    name: str = "Anonymous"
    connected_at: float = field(default_factory=time.time)


class ConnectionManager:
    """Manages WebSocket connections for all players."""

    def __init__(self) -> None:
        self.connections: Dict[str, PlayerConnection] = {}

    async def connect(self, websocket: WebSocket, player_id: str, name: str = "Anonymous") -> PlayerConnection:
        await websocket.accept()
        conn = PlayerConnection(
            player_id=player_id,
            websocket=websocket,
            name=name,
        )
        self.connections[player_id] = conn
        logger.info(f"Player connected: {name} ({player_id})")
        return conn

    def disconnect(self, player_id: str) -> None:
        if player_id in self.connections:
            name = self.connections[player_id].name
            del self.connections[player_id]
            logger.info(f"Player disconnected: {name} ({player_id})")

    async def send_to_player(self, player_id: str, message: dict) -> None:
        conn = self.connections.get(player_id)
        if conn and conn.websocket:
            try:
                await conn.websocket.send_json(message)
            except Exception:
                logger.warning(f"Failed to send to player {player_id}")

    async def broadcast(self, message: dict, exclude: Optional[Set[str]] = None) -> None:
        exclude = exclude or set()
        for pid, conn in self.connections.items():
            if pid not in exclude and conn.websocket:
                try:
                    await conn.websocket.send_json(message)
                except Exception:
                    pass

    async def broadcast_to_faction(self, faction_id: str, message: dict) -> None:
        for conn in self.connections.values():
            if conn.faction_id == faction_id and conn.websocket:
                try:
                    await conn.websocket.send_json(message)
                except Exception:
                    pass

    def get_online_players(self) -> List[dict]:
        return [
            {"player_id": c.player_id, "name": c.name, "faction_id": c.faction_id}
            for c in self.connections.values()
        ]


# ---------------------------------------------------------------------------
# Game Room
# ---------------------------------------------------------------------------


class GameRoom:
    """A game room that holds one game session."""

    def __init__(self, room_id: str, host_player_id: str, name: str = "Default Room"):
        self.room_id = room_id
        self.host_player_id = host_player_id
        self.name = name
        self.engine = GameEngine()
        self.started = False
        self.paused = False
        self.auto_turn_interval: float = 0  # 0 = manual turns
        self._auto_turn_task: Optional[asyncio.Task] = None
        self.manager = ConnectionManager()
        self.chat_history: List[dict] = []
        self.max_chat_history = 200

    async def start_game(self, faction_configs: List[dict], **kwargs: Any) -> dict:
        """Start a new game in this room."""
        self.engine.new_game(faction_configs, **kwargs)
        self.started = True

        # Assign factions to connected players
        factions = list(self.engine.state.factions.values())
        for i, conn in enumerate(self.manager.connections.values()):
            if i < len(factions) and factions[i].player_id == conn.player_id:
                conn.faction_id = factions[i].id

        await self.manager.broadcast({
            "type": "game_started",
            "state": self.engine.state.to_dict(),
        })
        return {"ok": True}

    async def process_turn(self) -> dict:
        """Process one game turn and broadcast results."""
        if not self.started:
            return {"ok": False, "error": "Game not started"}

        events = self.engine.process_turn()
        event_dicts = [e.to_dict() for e in events]

        await self.manager.broadcast({
            "type": "turn_processed",
            "turn": self.engine.state.turn,
            "season": self.engine.state.season.value,
            "year": self.engine.state.year,
            "events": event_dicts,
            "state": self.engine.state.to_dict(),
        })

        return {"ok": True, "turn": self.engine.state.turn, "events": event_dicts}

    async def handle_action(self, player_id: str, action: dict) -> dict:
        """Handle a player action."""
        conn = self.manager.connections.get(player_id)
        if not conn or not conn.faction_id:
            return {"ok": False, "error": "Not in a faction"}

        result = self.engine.submit_action(conn.faction_id, action)
        return result

    async def handle_ready(self, player_id: str) -> dict:
        """Mark player as ready and potentially process turn."""
        conn = self.manager.connections.get(player_id)
        if not conn or not conn.faction_id:
            return {"ok": False, "error": "Not in a faction"}

        all_ready = self.engine.mark_ready(conn.faction_id)
        await self.manager.broadcast({
            "type": "player_ready",
            "player_id": player_id,
            "faction_id": conn.faction_id,
            "all_ready": all_ready,
        })

        if all_ready:
            return await self.process_turn()
        return {"ok": True, "all_ready": False}

    def save(self, filename: Optional[str] = None) -> str:
        """Save game to file."""
        SAVE_DIR.mkdir(exist_ok=True)
        filename = filename or f"save_{self.room_id}_{self.engine.state.turn}.json"
        filepath = SAVE_DIR / filename
        self.engine.save_game(str(filepath))
        return str(filepath)

    def load(self, filepath: str) -> None:
        """Load game from file."""
        self.engine.load_game(filepath)
        self.started = True


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Civilization Online RPG",
        description="A player-hosted online grand strategy RPG inspired by RTK14 and EU4",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Game rooms
    rooms: Dict[str, GameRoom] = {}
    default_room_id = "default"

    def get_or_create_room(room_id: str = "default") -> GameRoom:
        if room_id not in rooms:
            rooms[room_id] = GameRoom(
                room_id=room_id,
                host_player_id="system",
                name=f"Room {room_id}",
            )
        return rooms[room_id]

    # -----------------------------------------------------------------------
    # Static files (frontend)
    # -----------------------------------------------------------------------

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return index_path.read_text(encoding="utf-8")
        return HTMLResponse("<h1>Civilization Online RPG Server</h1><p>Frontend not found.</p>")

    # -----------------------------------------------------------------------
    # REST API
    # -----------------------------------------------------------------------

    @app.get("/api/status")
    async def server_status():
        return {
            "status": "online",
            "rooms": len(rooms),
            "version": "0.1.0",
        }

    @app.get("/api/rooms")
    async def list_rooms():
        return {
            "rooms": [
                {
                    "id": r.room_id,
                    "name": r.name,
                    "started": r.started,
                    "players": len(r.manager.connections),
                    "turn": r.engine.state.turn if r.started else 0,
                }
                for r in rooms.values()
            ]
        }

    @app.post("/api/rooms/create")
    async def create_room(name: str = "New Room"):
        room_id = secrets.token_hex(4)
        room = GameRoom(room_id=room_id, host_player_id="system", name=name)
        rooms[room_id] = room
        return {"room_id": room_id, "name": name}

    @app.post("/api/rooms/{room_id}/start")
    async def start_game(room_id: str, config: dict = None):
        room = get_or_create_room(room_id)
        if room.started:
            return {"ok": False, "error": "Game already started"}

        # Default factions if no config provided
        if not config or "factions" not in config:
            players = room.manager.get_online_players()
            faction_configs = []
            colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
            for i, p in enumerate(players):
                faction_configs.append({
                    "name": f"{p['name']}'s Kingdom",
                    "color": colors[i % len(colors)],
                    "player_id": p["player_id"],
                })
            # Fill remaining slots with AI
            for i in range(len(players), 4):
                ai_names = ["Wei Empire", "Shu Kingdom", "Wu Dynasty", "Jin Alliance", "Yan State", "Chu Domain"]
                faction_configs.append({
                    "name": ai_names[i % len(ai_names)],
                    "color": colors[i % len(colors)],
                    "player_id": None,
                })
        else:
            faction_configs = config["factions"]

        map_cols = (config or {}).get("map_cols", 10)
        map_rows = (config or {}).get("map_rows", 8)
        seed = (config or {}).get("seed")

        result = await room.start_game(
            faction_configs, map_cols=map_cols, map_rows=map_rows, seed=seed
        )
        return result

    @app.get("/api/rooms/{room_id}/state")
    async def get_state(room_id: str):
        room = get_or_create_room(room_id)
        if not room.started:
            return {"ok": False, "error": "Game not started"}
        return room.engine.state.to_dict()

    @app.get("/api/rooms/{room_id}/tech_tree")
    async def get_tech_tree(room_id: str):
        tree = build_tech_tree()
        return {"techs": {k: v.to_dict() for k, v in tree.items()}}

    @app.get("/api/rooms/{room_id}/available_techs/{faction_id}")
    async def available_techs(room_id: str, faction_id: str):
        room = get_or_create_room(room_id)
        if not room.started:
            return {"ok": False, "error": "Game not started"}
        faction = room.engine.state.factions.get(faction_id)
        if not faction:
            return {"ok": False, "error": "Faction not found"}
        techs = get_available_techs(faction.researched_techs)
        return {"techs": [t.to_dict() for t in techs]}

    @app.post("/api/rooms/{room_id}/save")
    async def save_game(room_id: str, filename: Optional[str] = None):
        room = get_or_create_room(room_id)
        if not room.started:
            return {"ok": False, "error": "Game not started"}
        path = room.save(filename)
        return {"ok": True, "path": path}

    @app.post("/api/rooms/{room_id}/load")
    async def load_game(room_id: str, filepath: str = ""):
        room = get_or_create_room(room_id)
        if not filepath:
            return {"ok": False, "error": "No filepath provided"}
        try:
            room.load(filepath)
            return {"ok": True, "turn": room.engine.state.turn}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.get("/api/portrait/{character_name}")
    async def get_portrait(character_name: str, role: str = "free",
                           command: int = 50, force: int = 50,
                           intelligence: int = 50, politics: int = 50,
                           charisma: int = 50):
        svg = generate_portrait_svg(
            character_name, role=role,
            command=command, force=force,
            intelligence=intelligence, politics=politics,
            charisma=charisma,
        )
        return HTMLResponse(content=svg, media_type="image/svg+xml")

    @app.get("/api/rooms/{room_id}/portraits")
    async def get_all_portraits(room_id: str, faction_id: str = ""):
        room = get_or_create_room(room_id)
        if not room.started:
            return {"ok": False, "error": "Game not started"}
        portraits = {}
        for cid, char in room.engine.state.characters.items():
            if faction_id and char.faction_id != faction_id:
                continue
            if not char.alive:
                continue
            s = char.stats
            svg = generate_portrait_svg(
                char.name, role=char.role.value,
                command=s.command, force=s.force,
                intelligence=s.intelligence, politics=s.politics,
                charisma=s.charisma,
            )
            import base64
            encoded = base64.b64encode(svg.encode('utf-8')).decode('ascii')
            portraits[cid] = {
                "name": char.name,
                "role": char.role.value,
                "data_uri": f"data:image/svg+xml;base64,{encoded}",
            }
        return {"portraits": portraits}

    @app.get("/api/scenarios")
    async def get_scenarios():
        return {"scenarios": list_scenarios()}

    @app.get("/api/scenarios/{scenario_id}")
    async def get_scenario_detail(scenario_id: str):
        scenario = get_scenario(scenario_id)
        if not scenario:
            return {"ok": False, "error": "Scenario not found"}
        return {"ok": True, "scenario": scenario}

    @app.get("/api/saves")
    async def list_saves():
        SAVE_DIR.mkdir(exist_ok=True)
        saves = []
        for f in SAVE_DIR.glob("*.json"):
            saves.append({"filename": f.name, "size": f.stat().st_size})
        return {"saves": saves}

    # -----------------------------------------------------------------------
    # WebSocket
    # -----------------------------------------------------------------------

    @app.websocket("/ws/{room_id}")
    async def websocket_endpoint(
        websocket: WebSocket,
        room_id: str,
        player_id: str = Query(default=""),
        player_name: str = Query(default="Anonymous"),
    ):
        if not player_id:
            player_id = secrets.token_hex(8)

        room = get_or_create_room(room_id)
        conn = await room.manager.connect(websocket, player_id, player_name)

        # Send initial state
        init_msg: dict = {
            "type": "connected",
            "player_id": player_id,
            "room_id": room_id,
            "players": room.manager.get_online_players(),
        }
        if room.started:
            init_msg["state"] = room.engine.state.to_dict()
            # Find player's faction
            for f in room.engine.state.factions.values():
                if f.player_id == player_id:
                    conn.faction_id = f.id
                    init_msg["faction_id"] = f.id
                    break

        await room.manager.send_to_player(player_id, init_msg)

        # Notify others
        await room.manager.broadcast(
            {"type": "player_joined", "player_id": player_id, "name": player_name},
            exclude={player_id},
        )

        try:
            while True:
                data = await websocket.receive_json()
                await _handle_ws_message(room, conn, data)
        except WebSocketDisconnect:
            room.manager.disconnect(player_id)
            await room.manager.broadcast(
                {"type": "player_left", "player_id": player_id, "name": player_name}
            )
        except Exception as e:
            logger.error(f"WebSocket error for {player_id}: {e}")
            room.manager.disconnect(player_id)

    async def _handle_ws_message(room: GameRoom, conn: PlayerConnection, data: dict) -> None:
        """Handle incoming WebSocket message."""
        msg_type = data.get("type", "")

        if msg_type == "action":
            result = await room.handle_action(conn.player_id, data.get("action", {}))
            await room.manager.send_to_player(conn.player_id, {
                "type": "action_result", **result,
            })

        elif msg_type == "ready":
            result = await room.handle_ready(conn.player_id)
            if result.get("ok") and "turn" in result:
                pass  # Turn broadcast already handled in handle_ready

        elif msg_type == "chat":
            msg = {
                "type": "chat",
                "player_id": conn.player_id,
                "name": conn.name,
                "faction_id": conn.faction_id,
                "text": data.get("text", "")[:500],
                "timestamp": time.time(),
            }
            room.chat_history.append(msg)
            if len(room.chat_history) > room.max_chat_history:
                room.chat_history = room.chat_history[-room.max_chat_history:]
            await room.manager.broadcast(msg)

        elif msg_type == "get_state":
            if room.started:
                await room.manager.send_to_player(conn.player_id, {
                    "type": "state_update",
                    "state": room.engine.state.to_dict(),
                })

        elif msg_type == "select_faction":
            faction_id = data.get("faction_id", "")
            if faction_id in room.engine.state.factions:
                faction = room.engine.state.factions[faction_id]
                if faction.player_id is None or faction.player_id == conn.player_id:
                    faction.player_id = conn.player_id
                    conn.faction_id = faction_id
                    await room.manager.send_to_player(conn.player_id, {
                        "type": "faction_selected",
                        "faction_id": faction_id,
                        "faction": faction.to_dict(),
                    })

        elif msg_type == "force_turn":
            # Host-only: force process turn
            if conn.player_id == room.host_player_id or True:  # Allow any for now
                await room.process_turn()

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Run the game server."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn is required. Install with: pip install uvicorn[standard]")

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")
