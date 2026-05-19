#!/usr/bin/env python3
import bounds
import numpy as np

try:
    res
except NameError:
    res = bounds.gen_bounds("random_forest.npz")

bounds, test_loss = res

def to_latex_bold(df, digits=3):
    form = lambda x: "{:.{}f}\\%".format(x*100, digits)
    df_s = df.style.format(form)
    for row in df.index:
        col = df.loc[row].idxmin()
        m = "{:.{}f}".format(df.loc[row][col]*100, digits)
        for entry, col in zip(df.loc[row], df.columns):
            if m == "{:.{}f}".format(entry*100, digits):
                df_s = df_s.format(lambda x: "\\textbf{{{:.{}f}\\%}}".format(x*100, digits), subset=(row, col))
    print(df_s.to_latex())

to_latex_bold(bounds.set_index("name")[["lambda", "tnd", "bennett"]], 1)
to_latex_bold(test_loss.set_index("name")[["lambda", "tnd", "bennett"]], 3)

