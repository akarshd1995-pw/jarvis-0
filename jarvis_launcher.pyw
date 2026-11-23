"""Windowless launcher used by autostart: pythonw.exe jarvis_launcher.pyw"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.argv = [sys.argv[0]]

from jarvis.daemon import main  # noqa: E402

main()
