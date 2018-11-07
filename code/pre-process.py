import pandas as pd
import numpy as np
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import time
import pickle
from PIL import Image

def load_image(path):
    current = io.imread(path)
    if len(current.shape) >= 3:
        print("removing extra data!!")
        current = current[:,:,0]
    return current

def expand_with_channels(image_array):
    channel = []
    channel.append(current)
    channel.append(current)
    channel.append(current)
    return channel

def get_label(dataframe, image_name):
    d2 = dataframe[(dataframe['Image Index'] == image_name)]
    value = d2.iloc[0]['label']
    if value:
        return 1
    else:
        return 0


# dataframe = pd.read_csv("D:\\Data_Entry_2017.csv")
dataframe = pd.read_csv("../Data_Entry_2017.csv")


# labels = dataframe["Image Index"]
dataframe["label"] = dataframe["Finding Labels"].apply(lambda x: "Pneumonia" in x)

true_labels = dataframe.query('label==True')
false_labels = dataframe.query('label==False')



start_time = time.time()
i = 1
total = 1

data_set_name = "images_001"

path_in = "../" + data_set_name
path_out = "../data/train/"

data = []
labels = []
last = None

for iname in os.listdir(path_in):


    label = get_label(dataframe, iname)


    if label != last:
        current = load_image(path_in + "/" + iname)
        labels.append(label)
        data.append(expand_with_channels(current))
        total += 1
        last = label

    if total % 16 == 0:
        PATH_OUTPUT = path_out  + "/" + data_set_name + "_" + iname.split(".png")[0] + "/"
        os.makedirs(PATH_OUTPUT, exist_ok=True)
        np.save(PATH_OUTPUT + "data", np.array(data))
        np.save(PATH_OUTPUT + "labels", np.array(labels))
        data = []
        labels = []
        print(str(total))
        total = 1
    i = i + 1


end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))
