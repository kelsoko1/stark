"""
ws_bridge.py
WebSocket server — relays live training metrics to the dashboard.
Runs alongside trainer.py in a background thread.
"""

import asyncio
import json
import threading
import time
import websockets

# Import the shared emitter from trainer
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "train"))


HOST = "localhost"
PORT = 8765

CLIENTS = set()

async def handler(websocket):
    CLIENTS.add(websocket)
    print(f"[WS] Dashboard connected ({len(CLIENTS)} clients)")
    try:
        async for message in websocket:
            # Accept control signals from dashboard (pause/resume/stop)
            try:
                cmd = json.loads(message)
                print(f"[WS] Control: {cmd}")
            except Exception:
                pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        CLIENTS.discard(websocket)
        print(f"[WS] Dashboard disconnected ({len(CLIENTS)} clients)")


async def broadcast_loop():
    """Drains the metrics queue and broadcasts to all connected dashboards."""
    while True:
        await asyncio.sleep(0.5)
        if not CLIENTS:
            continue
        # Try to import EMITTER from trainer (if trainer is running)
        try:
            from trainer import EMITTER
            items = EMITTER.drain()
            for item in items:
                payload = json.dumps(item)
                dead = set()
                for ws in CLIENTS:
                    try:
                        await ws.send(payload)
                    except Exception:
                        dead.add(ws)
                CLIENTS -= dead
        except ImportError:
            pass


async def main():
    print(f"[WS] WebSocket server starting on ws://{HOST}:{PORT}")
    async with websockets.serve(handler, HOST, PORT):
        await broadcast_loop()


def run_ws_server():
    """Run in a background thread."""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
