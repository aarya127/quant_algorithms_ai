"""
conftest.py — pytest path & fixture setup for the full test suite.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "algorithms/machine_learning_algorithms/supervised"))
sys.path.insert(0, str(ROOT / "algorithms/machine_learning_algorithms/data_pipelines"))
