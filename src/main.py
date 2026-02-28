import os
import sys


# Ensure src/ is on the path regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from core.app import App


if __name__ == "__main__":
    App().run()
