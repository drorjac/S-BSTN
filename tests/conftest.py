import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# Tests reference configs/ relative to the repository root.
os.chdir(Path(__file__).resolve().parents[1])
