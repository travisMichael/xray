from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import time
from PIL import Image

# image = io.imread("D:\\x-ray\\images_004\\00006585_007.PNG")

i = 0
start_time = time.time()

# pc
# path_in = "D:\\x-ray\\images_001"
# path_out = "D:\\output\\images_001"

# mac
path_in = "/Users/a1406632/Downloads/Images_001"
path_out = "/Users/a1406632/data_train/images_train"

for iname in os.listdir(path_in):
    foo = Image.open(path_in + "/" + iname)
    foo = foo.resize((224, 224), Image.ANTIALIAS)
    foo.save(path_out + "\\" + iname, optimize=True, quality=95)
    i = i + 1
    print(str(i))

end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))
print('Image name: {}')