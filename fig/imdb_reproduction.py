#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

a = dict(np.load(Path("~/Downloads/pac-bayes-predictions/imdb_predictions.npz").expanduser()))

