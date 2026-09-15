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


def main():
    import uvicorn

    host = "127.0.0.1"
    port = 8000

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
