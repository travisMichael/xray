import os
import pandas as pd
from skimage import io, transform
import numpy as np
import time

true_processed_elements = 1
false_processed_elements = 1
false_data = []
true_data = []

def load_image(path):
    current = io.imread(path)
    if len(current.shape) >= 3:
        print("removing extra data!!")
        current = current[:,:,0]
    return current


def dataframe_function(row):
    global true_processed_elements
    global false_processed_elements
    global false_data
    global true_data



    if row.label == True:
        image_data = load_image("../" + row[1] + "/" + row[0])
        true_data.append(image_data)
        true_processed_elements = true_processed_elements + 1
    else:
        if false_processed_elements < 1100:
            image_data = load_image("../" + row[1] + "/" + row[0])
            false_data.append(image_data)
            false_processed_elements = false_processed_elements + 1

    if false_processed_elements % 1000 == 0:
        false_processed_elements = false_processed_elements + 1
        path = "../data/train/negative/"
        save(false_data, false_processed_elements, path)
        false_data = []

    if true_processed_elements % 200 == 0:
        true_processed_elements = true_processed_elements + 1
        path = "../data/train/positive/"
        save(true_data, true_processed_elements, path)
        true_data = []

def save(data, number_processed, path):
    PATH_OUTPUT = path
    os.makedirs(PATH_OUTPUT, exist_ok=True)
    np.save(PATH_OUTPUT + str(number_processed) + "original_data", np.array(data))
    print("saved " + path)



dataframe = pd.read_csv("../Data_Entry_2017.csv")


dataframe["label"] = dataframe["Finding Labels"].apply(lambda x: "Pneumonia" in x)

# group by "patient ID"


start_time = time.time()

path_out = "../data/train/"

image_df_001 = pd.DataFrame(os.listdir("../images_001"))
image_df_002 = pd.DataFrame(os.listdir("../images_002"))
image_df_003 = pd.DataFrame(os.listdir("../images_003"))
image_df_004 = pd.DataFrame(os.listdir("../images_004"))

image_df_001['data_set'] = "images_001"
image_df_002['data_set'] = "images_002"
image_df_003['data_set'] = "images_003"
image_df_004['data_set'] = "images_004"

frames = [image_df_001, image_df_002, image_df_003, image_df_004]
result = pd.concat(frames)
result["Image Index"] = result[result.columns[0]]

merged = pd.merge(result, dataframe, on="Image Index")
merged.apply(dataframe_function, axis=1)

path = "../data/train/positive/"
save(true_data, true_processed_elements, path)

path = "../data/train/negative/"
save(false_data, false_processed_elements, path)

end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))
