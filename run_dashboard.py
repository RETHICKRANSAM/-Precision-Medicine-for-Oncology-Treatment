"""
OncoNexus Dashboard Launcher.

Launches the FastAPI backend and serves the interactive Precision Oncology
Command Center at http://127.0.0.1:8000.
"""

import os
import sys
import webbrowser
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# Load root .env
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)


def print_banner(host: str, port: int):
    print("=" * 78)
    print("  ONCONEXUS: MULTI-STAGE PRECISION ONCOLOGY INTELLIGENCE COMMAND CENTER")
    print("=" * 78)
    print(f"  * Pipeline Architecture: Stage 01 (ML) -> 02 (DL) -> 03 (NLP) -> 04 (SLM) -> 05 (GenAI)")
    print(f"  * Database Storage:      Supabase Postgres (public.generated_scenarios)")
    print(f"  * Dashboard URL:         http://{host}:{port}")
    print(f"  * Interactive API Docs:  http://{host}:{port}/docs")
    print(f"  * Health Check:          http://{host}:{port}/api/health")
    print("=" * 78)
    print("  Press Ctrl+C to terminate the command center server.\n")


def is_port_available(host: str, port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="OncoNexus Command Center Launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    args = parser.parse_args()

    host = args.host
    port = args.port

    if not is_port_available(host, port):
        print(f"\n[!] Warning: Port {port} is currently in use or closing.")
        alt_port = port + 1
        while not is_port_available(host, alt_port) and alt_port < port + 10:
            alt_port += 1
        print(f"[*] Automatically binding to available port {alt_port} instead.\n")
        port = alt_port

    print_banner(host, port)

    # Launch Uvicorn
    uvicorn.run(
        "onconexu_api.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
