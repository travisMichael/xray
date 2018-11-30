import os
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn

from sklearn.metrics import *
from utils import train, evaluate, calculate_weigths
from plots import plot_learning_curves
from model import cnn, densenet
from loader import custom_data_loader


def train_model(model_type, dataset, path_to_data):
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    # Set a correct path to the seizure data file you downloaded
    PATH_TRAIN_FILE = path_to_data + "/train/"
    PATH_VALID_FILE = path_to_data + "/validation/"
    # PATH_TEST_FILE = "../data/test/"

    # Path for saving model
    PATH_OUTPUT = "../output/"
    os.makedirs(PATH_OUTPUT, exist_ok=True)

    # Some parameters
    NUM_EPOCHS = 10
    BATCH_SIZE = 8
    USE_CUDA = True  # Set 'True' if you want to use GPU
    NUM_WORKERS = 0  # Number of threads used by DataLoader. You can adjust this according to your machine spec.

    train_loader = custom_data_loader.XrayLoader(PATH_TRAIN_FILE)
    valid_loader = custom_data_loader.XrayLoader(PATH_VALID_FILE)
    # test_loader = custom_data_loader.XrayLoader(PATH_TEST_FILE)

    weights = calculate_weigths(path_to_data)

    # model = dnet.densenet121()
    if model_type == "cnn":
        model = cnn.CNN()
    if model_type == "densenet":
        model = densenet.densenet121()

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

    best_roc = 0
    for epoch in range(NUM_EPOCHS):

        train_loader.reset()
        valid_loader.reset()

        train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
        valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)

        roc = 0.0
        y1, y2 = zip(*valid_results)
        try:
            roc = sklearn.metrics.roc_auc_score(y1, y2)
        except:
            print("exception")
        print("made it")

        if roc > best_roc:
            best_roc = roc
            # torch.save(model, os.path.join(PATH_OUTPUT, 'cnn.pth'))
            torch.save(model, "../output/" + model_type + "_" + dataset + ".pth")



train_model("cnn", "original", "../data")

#
# plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

# test_loss, test_accuracy, test_results = evaluate(best_model, device, test_loader, criterion)
#
# class_names = ['Seizure', 'TumorArea', 'HealthyArea', 'EyesClosed', 'EyesOpen']
# plot_confusion_matrix(test_results, class_names)
