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

dataframe = pd.read_csv("D:\\Data_Entry_2017.csv")


# labels = dataframe["Image Index"]
dataframe["label"] = dataframe["Finding Labels"].apply(lambda x: "Pneumonia" in x)
# print("hello") # Pneumonia




start_time = time.time()
i = 1
total = 1
# input = torch.tensor(np.array(batch), dtype=torch.float)
path_in = "D:\\x-ray\\images_001"
path_out = "D:\\output\\images_001"

data = []
labels = []
last = None
for iname in os.listdir(path_out):

    current = io.imread(path_out + "\\" + iname)
    if len(current.shape) >= 3:
        print("removing extra data!!")
        current = current[:,:,0]

    d2 = dataframe[(dataframe['Image Index'] == iname)]
    value = d2.iloc[0]['label']
    if value:
        label = 1
    else:
        label = 0


    labels.append(label)
    channel = []
    channel.append(current)
    channel.append(current)
    channel.append(current)
    data.append(channel)

    if i % 16 == 0:
        # d = np.array(data)
        # l = np.array(labels)
        np.save("data\\train_" + str(total), np.array(data))
        np.save("data\\labels_" + str(total), np.array(labels))
        data = []
        labels = []
        total += 1
    i = i + 1
    print(str(i))

input = np.array(data)
# np.save("aaa", data)
# with open('train.pkl', 'wb') as f:
#     pickle.dump(input, f)
end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))

# input = torch.tensor(np.array(batch), dtype=torch.float)
# dataset = TensorDataset(data, target)