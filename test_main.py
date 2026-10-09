"""Test script for main.py CLI mode with automated inputs."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Monkey-patch input() to feed automated queries
test_inputs = iter([
    "What time is it?",
    "What is today's date?",
    "Tell me about machine learning",
    "exit",
])

import builtins
_original_input = builtins.input
def mock_input(prompt=""):
    try:
        val = next(test_inputs)
        print(f"{prompt}{val}")
        return val
    except StopIteration:
        return "exit"

builtins.input = mock_input

# Now run main.py
from main import main
main()
