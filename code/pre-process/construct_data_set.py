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

def apply_validation(row):
    dataframe_function(row, "../data/validation/", "/Users/a1406632/Downloads/")

def apply_train(row):
    dataframe_function(row, "../data/train/", "../")

def dataframe_function(row, path_to_save, path_to_load):
    global true_processed_elements
    global false_processed_elements
    global false_data
    global true_data

    if row.label == True:
        image_data = load_image(path_to_load + row[1] + "/" + row[0])
        true_data.append(image_data)
        true_processed_elements = true_processed_elements + 1
    else:
        if false_processed_elements < 350:
            image_data = load_image(path_to_load + row[1] + "/" + row[0])
            false_data.append(image_data)
            false_processed_elements = false_processed_elements + 1

    if false_processed_elements % 300 == 0:
        false_processed_elements = false_processed_elements + 1
        path = path_to_save + "negative/"
        save(false_data, false_processed_elements, path)
        false_data = []

    if true_processed_elements % 200 == 0:
        true_processed_elements = true_processed_elements + 1
        path = path_to_save + "positive/"
        save(true_data, true_processed_elements, path)
        true_data = []

def save(data, number_processed, path):
    PATH_OUTPUT = path
    os.makedirs(PATH_OUTPUT, exist_ok=True)
    np.save(PATH_OUTPUT + str(number_processed) + "original_data", np.array(data))
    print("saved " + path)



def construct_train(dataframe):
    start_time = time.time()

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
    merged.apply(apply_train, axis=1)

    path = "../data/train/positive/"
    save(true_data, true_processed_elements, path)

    path = "../data/train/negative/"
    save(false_data, false_processed_elements, path)

    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
    print("train")

def construct_validation(dataframe):
    start_time = time.time()

    image_df_005 = pd.DataFrame(os.listdir("/Users/a1406632/Downloads/images_005"))
    image_df_006 = pd.DataFrame(os.listdir("/Users/a1406632/Downloads/images_006"))
    image_df_007 = pd.DataFrame(os.listdir("/Users/a1406632/Downloads/images_007"))
    image_df_008 = pd.DataFrame(os.listdir("/Users/a1406632/Downloads/images_008"))

    image_df_005['data_set'] = "images_005"
    image_df_006['data_set'] = "images_006"
    image_df_007['data_set'] = "images_007"
    image_df_008['data_set'] = "images_008"

    frames = [image_df_005, image_df_006, image_df_007, image_df_008]
    result = pd.concat(frames)
    result["Image Index"] = result[result.columns[0]]

    merged = pd.merge(result, dataframe, on="Image Index")
    merged.apply(apply_validation, axis=1)

    path = "../data/validation/positive/"
    save(true_data, true_processed_elements, path)

    path = "../data/validation/negative/"
    save(false_data, false_processed_elements, path)

    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
    print("train")
    print("train")



dataframe = pd.read_csv("../Data_Entry_2017.csv")


dataframe["label"] = dataframe["Finding Labels"].apply(lambda x: "Pneumonia" in x)

# construct_validation(dataframe)

construct_train(dataframe)