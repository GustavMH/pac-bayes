#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from itertools import islice

def sort_chroma(folder: path):
    def c(im):
        return im[:,:,0].mean() / im[:,:,1].mean()

    fs = list(islice(folder.glob("*.jpg"), 30))
    imgs = [np.array(Image.open(f)) for f in fs]
    chroma = [c(im) for im in imgs]
    print(np.array(sorted(chroma)))
    return [x for _, x in sorted(zip(chroma, imgs))]

folders = list(Path("~/Downloads/").expanduser().glob("EuroSAT*/*"))
folder_names = [f.stem for f in folders]
chroma = np.array([sort_chroma(f) for f in folders])

arr = np.hstack(np.hstack(chroma))
Image.fromarray(arr).save("dist_shift_ill.jpg")
