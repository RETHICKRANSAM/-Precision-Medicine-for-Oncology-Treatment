"""
Supabase Client Initializer for GenAI Compound Scenario Generator.

Handles secure authentication, environment variable loading, and connectivity checks.
Never logs or exposes secret credentials.
"""

import os
import re
from typing import Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
from supabase._sync.client import SupabaseException


def load_supabase_credentials() -> Tuple[str, str]:
    """
    Loads and validates Supabase credentials from environment variables.
    Searches root .env file and active environment.
    
    Returns:
        Tuple of (supabase_url, supabase_key)
        
    Raises:
        RuntimeError with descriptive error message if missing or invalid.
    """
    # 1. Locate .env file
    env_path = Path(".env")
    if not env_path.exists():
        # Check parent directory if running from a subdirectory
        parent_env = Path(__file__).resolve().parents[2] / ".env"
        if parent_env.exists():
            env_path = parent_env

    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        # Check if environment variables already exist in system env
        if not os.getenv("SUPABASE_URL") or not (os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")):
            raise RuntimeError(
                "Error: .env configuration file not found. "
                "Please create a .env file in the project root with SUPABASE_URL and SUPABASE_KEY."
            )

    # 2. Validate SUPABASE_URL
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    if not supabase_url:
        raise RuntimeError(
            "Error: Missing environment variable 'SUPABASE_URL'. "
            "Please define SUPABASE_URL in your .env file (e.g., https://<project-id>.supabase.co)."
        )

    if not re.match(r"^https?://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(/.*)?$", supabase_url):
        raise RuntimeError(
            f"Error: Invalid 'SUPABASE_URL' format. Expected standard HTTPS URL."
        )

    # 3. Validate SUPABASE_KEY
    supabase_key = os.getenv("SUPABASE_KEY", "").strip()
    if not supabase_key:
        # Fallback to SUPABASE_ANON_KEY if defined
        supabase_key = os.getenv("SUPABASE_ANON_KEY", "").strip()

    if not supabase_key:
        raise RuntimeError(
            "Error: Missing environment variable 'SUPABASE_KEY'. "
            "Please define SUPABASE_KEY in your .env file."
        )

    # If modern publishable key format is passed to supabase-py, fallback to anon JWT if available
    if supabase_key.startswith("sb_publishable_"):
        anon_key = os.getenv("SUPABASE_ANON_KEY", "").strip()
        if anon_key and re.match(r"^[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*$", anon_key):
            supabase_key = anon_key
        else:
            raise RuntimeError(
                "Error: Python supabase client requires a valid JWT API key (anon key) for REST operations. "
                "Please set SUPABASE_KEY to your project's anon JWT key in .env."
            )

    return supabase_url, supabase_key


def get_supabase_client() -> Client:
    """
    Initializes and returns an authenticated official Supabase Python Client.
    """
    url, key = load_supabase_credentials()
    try:
        client = create_client(url, key)
        return client
    except SupabaseException as exc:
        raise RuntimeError(f"Error initializing Supabase client: {exc}")
    except Exception as exc:
        raise RuntimeError(f"Unexpected connection error initializing Supabase client: {exc}")


def test_connection(table_name: str = "generated_scenarios") -> Tuple[bool, str]:
    """
    Tests active connectivity to the Supabase database and verifies table access.
    
    Args:
        table_name: The table to query for read accessibility.
        
    Returns:
        Tuple of (is_successful, status_message)
    """
    try:
        client = get_supabase_client()
        # Query row count on target table
        response = client.table(table_name).select("count", count="exact").limit(1).execute()
        count = response.count if response.count is not None else 0
        return True, f"Successfully connected to Supabase. Table '{table_name}' accessible (current rows: {count})."
    except SupabaseException as exc:
        err_msg = str(exc)
        if "404" in err_msg or "relation" in err_msg.lower() or "not found" in err_msg.lower():
            return False, f"Table Error: Table '{table_name}' not found in public schema. Run migrations first."
        if "401" in err_msg or "403" in err_msg or "jwt" in err_msg.lower() or "policy" in err_msg.lower() or "row-level security" in err_msg.lower():
            return False, f"Permission Error: Row-Level Security (RLS) or permission denied on table '{table_name}': {exc}"
        return False, f"Supabase Error: {exc}"
    except RuntimeError as exc:
        return False, str(exc)
    except Exception as exc:
        err_str = str(exc)
        if "relation" in err_str.lower() or "does not exist" in err_str.lower() or "not found" in err_str.lower():
            return False, f"Table Error: Table '{table_name}' does not exist in public schema."
        if "policy" in err_str.lower() or "permission" in err_str.lower() or "401" in err_str or "403" in err_str:
            return False, f"Permission / RLS Error accessing table '{table_name}': {exc}"
        return False, f"Connection Failed: {exc}"


if __name__ == "__main__":
    print("Testing Supabase connection from environment...")
    success, message = test_connection()
    status_label = "SUCCESS" if success else "FAILED"
    print(f"[{status_label}] {message}")
