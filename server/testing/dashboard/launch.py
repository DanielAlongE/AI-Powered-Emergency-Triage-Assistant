#!/usr/bin/env python3
"""
Launch script for the Streamlit dashboard.
"""
import sys
import subprocess
from pathlib import Path


def main():
    """Launch the Streamlit dashboard."""
    dashboard_path = Path(__file__).parent / "app.py"

    print("Launching Emergency Triage Agent Dashboard...")
    print(f"Dashboard path: {dashboard_path}")
    print("Opening in browser...")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light",
            "--theme.primaryColor", "#2E86AB"
        ])
    except KeyboardInterrupt:
        print("\nDashboard stopped")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")


if __name__ == "__main__":
    main()