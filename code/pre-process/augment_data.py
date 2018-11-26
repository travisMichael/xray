import os
import numpy as np
import time
import salt_pepper_transform as spt


# flow
# group images by patients
# split training set by patients (train, validation, test)
# training..
#   split images with positive labels and negative labels
#   batch size = 16 (8 positive, 8 negative)
# try to load next batch in a new thread
# load while there are negative labels to train
#    load next 8 negative labels
#    load next 8 positive labels, if all have been iterated, move to next transform and move



def transform_and_save(file):
    np_data = np.load(file)
    transformed_data = []
    transformed_file = file.split("original_data.npy")[0] + "salt_pepper_data"
    for i in range(np_data.shape[0]):
        transformed_data.append(spt.salt_and_pepper(np_data[i]))

    save(transformed_data, transformed_file)

def save(data, path):
    PATH_OUTPUT = path
    np.save(PATH_OUTPUT, np.array(data))
    print("saved " + path)

start_time = time.time()

path = "../../data/train/positive/"

for bundled in os.listdir(path):
    if "original" in bundled:
        transform_and_save(path + bundled)


end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))
