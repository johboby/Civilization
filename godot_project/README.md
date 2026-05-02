# Civilization Online RPG - Godot 4 Client

A complete grand strategy RPG game built with **Godot Engine 4.2**, inspired by **Romance of the Three Kingdoms 14** and **Europa Universalis 4**.

## Requirements

- [Godot Engine 4.2+](https://godotengine.org/download) (standard or .NET version)

## How to Run

1. Download and install Godot 4.2+
2. Open Godot and import this project (select the `project.godot` file)
3. Click the Play button or press F5

## Project Structure

```
godot_project/
  project.godot                    # Godot project configuration
  icon.svg                         # Project icon
  scenes/
    main_menu/main_menu.tscn       # Main menu scene
    game_world/game_world.tscn     # Game world with hex map and HUD
  scripts/
    autoload/                      # Global singletons (autoloads)
      game_data.gd                 # Global game state, data models, queries
      event_bus.gd                 # Signal bus for decoupled communication
      network_manager.gd           # WebSocket multiplayer networking
      audio_manager.gd             # Audio management (placeholder)
    game/                          # Core game logic
      map_generator.gd             # Hex-grid province map generation
      tech_tree.gd                 # Technology tree (27 techs, 4 branches)
      turn_processor.gd            # Turn processing (economy, combat, diplomacy)
      game_setup.gd                # New game initialization
      hex_map.gd                   # Hex map rendering and interaction
    ui/                            # UI scripts
      main_menu.gd                 # Main menu logic
      game_world.gd                # Game world controller (camera, actions)
      game_hud.gd                  # Top bar, event log, chat
      side_panel.gd                # Tabbed side panel (province, army, etc.)
```

## Game Features

### Map & Provinces
- Hex-grid map with 7 terrain types (Plains, Mountains, Forest, Desert, River, Coast, Marsh)
- Terrain affects movement speed, combat bonuses, and resource production
- Province development, buildings, fortifications, and unrest

### Characters (RTK14-Style)
- 5 core stats: Command, Force, Intelligence, Politics, Charisma
- Roles: Ruler, General, Strategist, Governor, Diplomat, Spy
- Aging, death, succession, loyalty, and defection mechanics
- Experience and leveling system

### Economy (EU4-Style)
- Three resources: Gold, Food, Manpower
- Tax rates, trade income, army upkeep
- 8 building types with upgrades
- Seasonal effects (Spring bonus, Winter penalties)

### Military
- Army recruitment with Infantry/Cavalry/Archer composition
- Province-to-province movement with terrain speed modifiers
- Tactical combat with general bonuses and terrain advantages
- Siege mechanics for capturing enemy provinces

### Technology
- 4 branches: Military, Economy, Culture, Diplomacy
- 27 technologies with prerequisites
- Academy buildings boost research speed

### Diplomacy
- Alliance, War, Trade, Truce relationships
- Opinion system with natural drift
- War exhaustion and stability effects

### Multiplayer
- WebSocket connection to the Python game server
- Join games hosted by other players
- Real-time chat
- Works with the `online_rpg` Python server in the parent directory

## Controls

| Key / Action | Description |
|-------------|-------------|
| W/A/S/D | Pan camera |
| Mouse Scroll | Zoom in/out |
| Middle Mouse Drag | Pan camera |
| Left Click | Select province |
| T | Toggle tech panel |
| F | Toggle diplomacy panel |
| Enter | End turn |

## Multiplayer Setup

### Single Player
Click "Single Player" on the main menu. All game logic runs locally.

### Online Multiplayer
1. Start the Python server: `python run_server.py` (from parent directory)
2. In Godot client, click "Join Online Game"
3. Enter the server URL (e.g., `ws://192.168.1.100:8000`)
4. Share the server URL with other players

## License

MIT License
