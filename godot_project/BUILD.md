# Building for Windows

## Prerequisites

1. Download [Godot 4.2+](https://godotengine.org/download/archive/) (Standard version)
2. Download the [Windows Desktop Export Template](https://godotengine.org/download/archive/) matching your Godot version

## Build Steps

### Option 1: Using Godot Editor

1. Open the project in Godot 4.2+
2. Go to `Project > Export...`
3. Select "Windows Desktop" preset
4. Click "Export Project..."
5. Choose output location and click "Save"
6. The executable will be at `builds/windows/CivilizationRPG.exe`

### Option 2: Command Line Build

```bash
# Set your Godot path
GODOT_PATH="/path/to/Godot_v4.2-stable_win64.exe"

# Export for Windows
$GODOT_PATH --headless --export-release "Windows Desktop" builds/windows/CivilizationRPG.exe

# Export for Linux
$GODOT_PATH --headless --export-release "Linux" builds/linux/CivilizationRPG.x86_64
```

### Option 3: GitHub Actions (CI/CD)

Create `.github/workflows/build.yml` in your repo:

```yaml
name: Build Game
on: [push, workflow_dispatch]
jobs:
  export:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: chickensoft-games/setup-godot@v2
        with:
          version: 4.2.2
      - name: Export Windows
        run: |
          cd godot_project
          godot --headless --export-release "Windows Desktop" ../builds/CivilizationRPG.exe
      - uses: actions/upload-artifact@v4
        with:
          name: windows-build
          path: builds/
```

## Distribution

After building, distribute the following files together:
- `CivilizationRPG.exe` (or `.x86_64` for Linux)
- `CivilizationRPG.pck` (if not embedded)

Players just double-click the exe to play. No installation required.

## Multiplayer Setup

For multiplayer, one player needs to run the Python server:
```bash
pip install fastapi uvicorn websockets
python run_server.py --host 0.0.0.0
```

Other players connect via the Godot client's "Join Online Game" option.
