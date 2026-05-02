#!/usr/bin/env python3
"""Launch the Civilization Online RPG server.

Usage:
    python run_server.py                    # Default: localhost:8000
    python run_server.py --port 9000        # Custom port
    python run_server.py --host 0.0.0.0     # Listen on all interfaces (LAN play)

Players connect via browser at http://<server-ip>:<port>
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Civilization Online RPG Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_server.py                     Start on localhost:8000
  python run_server.py --port 9000         Custom port
  python run_server.py --host 0.0.0.0      LAN accessible

Free hosting options for players:
  - Replit (https://replit.com)
  - Railway (https://railway.app)
  - Render (https://render.com)
  - Fly.io (https://fly.io)
  - Oracle Cloud Free Tier
  - Google Cloud Free Tier
        """,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn is required. Install with:")
        print("  pip install uvicorn[standard]")
        sys.exit(1)

    try:
        from online_rpg.server import create_app
    except ImportError:
        # Try adding parent dir to path
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from online_rpg.server import create_app

    print("=" * 50)
    print("  Civilization Online RPG Server")
    print("=" * 50)
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  URL:  http://{'localhost' if args.host == '0.0.0.0' else args.host}:{args.port}")
    print("=" * 50)
    print()

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
