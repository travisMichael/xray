import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
import numpy as np
import os
from sklearn.metrics import *
from utils import train, evaluate, calculate_weigths
from model import cnn, densenet
from loader import custom_data_loader


def train_model(type, dataset, path):
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    # Set a correct path to the seizure data file you downloaded
    PATH_TRAIN_FILE = path + "/data/train/"
    PATH_VALID_FILE = path + "/data/validation/"
    # PATH_TEST_FILE = "../data/test/"

    # Path for saving model
    # PATH_OUTPUT = "../output/"
    # os.makedirs(PATH_OUTPUT, exist_ok=True)

    # Some parameters
    NUM_EPOCHS = 1
    BATCH_SIZE = 8
    USE_CUDA = True  # Set 'True' if you want to use GPU
    NUM_WORKERS = 0  # Number of threads used by DataLoader. You can adjust this according to your machine spec.

    NEGATIVE_BATCH_SIZE = 50
    POSITIVE_BATCH_SIZE = 4
    TOTAL = NEGATIVE_BATCH_SIZE + POSITIVE_BATCH_SIZE


    if "original" != dataset and type == "additive_augmentation":
        POSITIVE_BATCH_SIZE = POSITIVE_BATCH_SIZE * 2
    weights = [(float(POSITIVE_BATCH_SIZE) / TOTAL),(float(NEGATIVE_BATCH_SIZE) / TOTAL)]

    model = cnn.CNN()

    # print("saving " + path + " " + dataset)
    # torch.save(model, path + "/output/" + model_type + "_" + dataset + ".pth")

    train_loader = custom_data_loader.XrayLoader(PATH_TRAIN_FILE,
                                                 dataset=dataset,
                                                 augmentation=type,
                                                 negative_batch_size=NEGATIVE_BATCH_SIZE,
                                                 positive_batch_size=POSITIVE_BATCH_SIZE)
    valid_loader = custom_data_loader.XrayLoader(PATH_VALID_FILE,
                                                 negative_batch_size=NEGATIVE_BATCH_SIZE,
                                                 positive_batch_size=POSITIVE_BATCH_SIZE)
    # test_loader = custom_data_loader.XrayLoader(PATH_TEST_FILE)



    class_weights = torch.FloatTensor(weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    # BCELoss
    optimizer = optim.Adam(model.parameters())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model.to(device)
    criterion.to(device)

    best_val_acc = 0.0
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []

    best_roc = 0.0
    roc_list = []
    total_train = 0.0
    total_load = 0.0
    avg_batch_train_time = 0.0
    avg_batch_load_time = 0.0
    index = 0
    for epoch in range(NUM_EPOCHS):
        index += 1
        train_loader.reset()
        valid_loader.reset()

        _, _, tt, tl, avg_tt, avg_lt = train(model, device, train_loader, criterion, optimizer, epoch)
        valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)

        total_train += tt
        total_load += tl
        avg_batch_train_time += avg_tt
        avg_batch_load_time += avg_lt

        roc = 0.0
        y1, y2 = zip(*valid_results)
        try:
            roc = sklearn.metrics.roc_auc_score(y1, y2)
        except:
            print("exception")
        roc_list.append(roc)
        print("roc: " + str(roc))

        if roc > best_roc:
            best_roc = roc
            # torch.save(model, os.path.join(PATH_OUTPUT, 'cnn.pth'))
            print("saving " + type + " " + dataset)
            path_to_save = path + "/output/" + type + "/cnn" + "_" + dataset
            os.makedirs(path_to_save, exist_ok=True)
            torch.save(model, path_to_save + ".pth")

    avg_batch_train_time = avg_batch_train_time / index
    avg_batch_load_time = avg_batch_load_time / index
    return best_roc, roc_list, total_train, total_load, avg_batch_train_time, avg_batch_load_time


def process(dataset, type):
    PATH_OUTPUT = "../" + "/output/" + type + "/cnn" + "_" + dataset
    os.makedirs(PATH_OUTPUT, exist_ok=True)
    _, roc_list, tt, tl, avg_t_t, avg_l_t = train_model(type,dataset, "../")
    np.save(PATH_OUTPUT, roc_list)
    np.save(PATH_OUTPUT + "_time", np.array([tt, tl, avg_t_t, avg_l_t]))
    return str(tt) + " " + str(tl) + " " + str(avg_t_t) + " " + str(avg_l_t)

dataset = "original"
type = "additive_augmentation"

s = ""
s += process(dataset, type) + "\n"

dataset = "salt_and_pepper"
s += process(dataset, type) + "\n"

dataset = "reflection"
s += process(dataset, type) + "\n"

dataset = "rotation"
s += process(dataset, type) + "\n"

type = "random_augmentation"
dataset = "salt_and_pepper"
s += process(dataset, type) + "\n"

dataset = "reflection"

s += process(dataset, type) + "\n"

dataset = "rotation"
s += process(dataset, type) + "\n"

dataset = "all"
s += process(dataset, type) + "\n"

print(s)

# PATH_OUTPUT = "../" + "/output/" + type + "/cnn" + "_" + dataset
# os.makedirs(PATH_OUTPUT, exist_ok=True)
# _, a_original_roc_list, a_original_tt, a_original_tl, a_original_avg_t_t, a_original_avg_l_t = train_model(type, dataset, "../")
# np.save(PATH_OUTPUT, a_original_roc_list)
# np.save(PATH_OUTPUT + "_time", np.array([a_original_tt, a_original_tl, a_original_avg_t_t, a_original_avg_l_t]))


#
# type = "random_augmentation"
# PATH_OUTPUT = "../" + "/output/" + type + "/cnn" + "_" + dataset
# os.makedirs(PATH_OUTPUT, exist_ok=True)
# _, r_original_roc_list, r_original_tt, r_original_tl, r_original_avg_t_t, r_original_avg_l_t = train_model(type, dataset, "../")
# np.save(PATH_OUTPUT, r_original_roc_list)
# np.save(PATH_OUTPUT + "_time", np.array([r_original_tt, r_original_tl, r_original_avg_t_t, r_original_avg_l_t]))
#
#
# PATH_OUTPUT = "../" + "/output/" + type + "/cnn" + "_" + dataset
# os.makedirs(PATH_OUTPUT, exist_ok=True)
# _, a_original_roc_list, a_original_tt, a_original_tl, a_original_avg_t_t, a_original_avg_l_t = train_model(type, dataset, "../")
# np.save(PATH_OUTPUT, a_original_roc_list)
# np.save(PATH_OUTPUT + "_time", np.array([a_original_tt, a_original_tl, a_original_avg_t_t, a_original_avg_l_t]))
#
#
#
#
#
# print(str(a_original_tt) + " " + str(a_original_tl) + " " + str(a_original_avg_t_t) + " " + str(a_original_avg_l_t))
# print(str(r_original_tt) + " " + str(r_original_tl) + " " + str(r_original_avg_t_t) + " " + str(r_original_avg_l_t))