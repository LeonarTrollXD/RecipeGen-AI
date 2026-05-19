"""
Punto de entrada principal de la aplicación.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.view.cli_view import run_cli

if __name__ == "__main__":
    print("🚀 Iniciando Recipe Gen AI...")
    run_cli()