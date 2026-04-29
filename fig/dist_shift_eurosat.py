#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = Path("~/Downloads/eurosat_chroma_shift.npz").expanduser()
res = dict(np.load(path))

# (10 runs, 5 training mixes, 2 test sets, 15 snapshots, 2700 examples, 10 categories)
