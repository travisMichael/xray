import os
import pandas as pd
from skimage import io, transform
import numpy as np
import time
import random

train_positive_data = []
valid_positive_data = []
test_positive_data = []
train_negative_data = []
valid_negative_data = []
test_negative_data = []

train_positive_processed_elements = 0
valid_positive_processed_elements = 0
test_positive_processed_elements = 0
train_negative_processed_elements = 0
valid_negative_processed_elements = 0
test_negative_processed_elements = 0


def load_image(path):
    current = io.imread(path)
    if len(current.shape) >= 3:
        print("removing extra data!!")
        current = current[:,:,0]
    return current


def process_data(row, data, number_of_elements, path_to_save):
    path_to_load = "/Users/a1406632/Downloads/"

    if number_of_elements > 201:
        return data, number_of_elements
    image_data = load_image(path_to_load + row[1] + "/" + row[0])

    data.append(image_data)
    number_of_elements += 1

    if number_of_elements % 50 == 0:
        path = "../data/" + path_to_save
        save(data, number_of_elements, path)
        data = []
    return data, number_of_elements


def dataframe_function(row):
    global train_positive_data
    global valid_positive_data
    global test_positive_data
    global train_negative_data
    global valid_negative_data
    global test_negative_data
    global train_positive_processed_elements
    global valid_positive_processed_elements
    global test_positive_processed_elements
    global train_negative_processed_elements
    global valid_negative_processed_elements
    global test_negative_processed_elements


    if row.label == True:
        if row.test_set == "train":
            train_positive_data, train_positive_processed_elements = process_data(row, train_positive_data, train_positive_processed_elements, "train/positive/")
        elif row.test_set == "validation":
            valid_positive_data, valid_positive_processed_elements = process_data(row, valid_positive_data, valid_positive_processed_elements, "validation/positive/")
        else:
            test_positive_data, test_positive_processed_elements = process_data(row, test_positive_data, test_positive_processed_elements, "test/positive/")
    else:
        if row.test_set == "train":
            train_negative_data, train_negative_processed_elements = process_data(row, train_negative_data, train_negative_processed_elements, "train/negative/")
        elif row.test_set == "validation":
            valid_negative_data, valid_negative_processed_elements = process_data(row, valid_negative_data, valid_negative_processed_elements, "validation/negative/")
        else:
            test_negative_data, test_negative_processed_elements = process_data(row, test_negative_data, test_negative_processed_elements, "test/negative/")




def save(data, number_processed, path):
    PATH_OUTPUT = path
    os.makedirs(PATH_OUTPUT, exist_ok=True)
    np.save(PATH_OUTPUT + str(number_processed) + "original_data", np.array(data))
    print("saved " + path + " " + str(number_processed))



def construct(dataframe):
    start_time = time.time()

    image_df_001 = pd.DataFrame(os.listdir("/Users/a1406632/Downloads/images_001"))
    image_df_002 = pd.DataFrame(os.listdir("/Users/a1406632/Downloads/images_002"))
    image_df_003 = pd.DataFrame(os.listdir("/Users/a1406632/Downloads/images_003"))
    image_df_004 = pd.DataFrame(os.listdir("/Users/a1406632/Downloads/images_004"))
    image_df_005 = pd.DataFrame(os.listdir("/Users/a1406632/Downloads/images_005"))
    image_df_006 = pd.DataFrame(os.listdir("/Users/a1406632/Downloads/images_006"))
    image_df_007 = pd.DataFrame(os.listdir("/Users/a1406632/Downloads/images_007"))
    image_df_008 = pd.DataFrame(os.listdir("/Users/a1406632/Downloads/images_008"))

    image_df_001['data_set'] = "images_001"
    image_df_002['data_set'] = "images_002"
    image_df_003['data_set'] = "images_003"
    image_df_004['data_set'] = "images_004"
    image_df_005['data_set'] = "images_005"
    image_df_006['data_set'] = "images_006"
    image_df_007['data_set'] = "images_007"
    image_df_008['data_set'] = "images_008"

    frames = [image_df_001, image_df_002, image_df_003, image_df_004, image_df_005, image_df_006, image_df_007, image_df_008]
    result = pd.concat(frames)
    result["Image Index"] = result[result.columns[0]]

    merged = pd.merge(result, dataframe, on="Image Index")
    merged.apply(dataframe_function, axis=1)

    save(train_positive_data, train_positive_processed_elements, "../data/train/positive/")
    save(valid_positive_data, valid_positive_processed_elements, "../data/validation/positive/")
    save(test_positive_data, test_positive_processed_elements, "../data/test/positive/")

    save(train_negative_data, train_negative_processed_elements, "../data/train/negative/")
    save(valid_negative_data, valid_negative_processed_elements, "../data/validation/negative/")
    save(test_negative_data, test_negative_processed_elements, "../data/test/negative/")

    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))

def apply_assign_to_test_set(row):
    rand = random.randint(1,100)
    if rand >= 0 and rand <= 70:
        return "train"
    elif rand <= 80:
        return "validation"
    else:
        return "test"

dataframe = pd.read_csv("../Data_Entry_2017.csv")


dataframe["label"] = dataframe["Finding Labels"].apply(lambda x: "Pneumonia" in x)
dataframe["test_set"] = dataframe["Finding Labels"].apply(apply_assign_to_test_set)



construct(dataframe)