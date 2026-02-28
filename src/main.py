"""Entry point — run from the project root: python src/main.py."""

import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.core.app import App


if __name__ == "__main__":
    App().run()
