import os
import torch
import torch.nn as nn
import torch.optim as optim

from utils import train, evaluate
from plots import plot_learning_curves
from model import cnn
from loader import custom_data_loader

torch.manual_seed(0)
if torch.cuda.is_available():
	torch.cuda.manual_seed(0)

# Set a correct path to the seizure data file you downloaded
PATH_TRAIN_FILE = "../data/train/"
PATH_VALID_FILE = "../data/validation/"
PATH_TEST_FILE = "../data/test/"

# Path for saving model
PATH_OUTPUT = "../output/"
os.makedirs(PATH_OUTPUT, exist_ok=True)

# Some parameters
NUM_EPOCHS = 10
BATCH_SIZE = 8
USE_CUDA = True  # Set 'True' if you want to use GPU
NUM_WORKERS = 0  # Number of threads used by DataLoader. You can adjust this according to your machine spec.

# train_dataset = load_seizure_dataset(PATH_TRAIN_FILE, MODEL_TYPE)
# valid_dataset = load_seizure_dataset(PATH_VALID_FILE, MODEL_TYPE)
# test_dataset = load_seizure_dataset(PATH_TEST_FILE, MODEL_TYPE)
# XrayLoader
train_loader = custom_data_loader.XrayLoader(PATH_TRAIN_FILE)
valid_loader = custom_data_loader.XrayLoader(PATH_VALID_FILE)
test_loader = custom_data_loader.XrayLoader(PATH_TEST_FILE)

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
# valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# model = dnet.densenet121()
model = cnn.CNN()
weights = [0.45, 0.55]
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

for epoch in range(NUM_EPOCHS):
	train_loader.reset()
	valid_loader.reset()

	train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
	valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)

	train_losses.append(train_loss)
	valid_losses.append(valid_loss)

	train_accuracies.append(train_accuracy)
	valid_accuracies.append(valid_accuracy)



plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)


# test_loss, test_accuracy, test_results = evaluate(best_model, device, test_loader, criterion)

class_names = ['Seizure', 'TumorArea', 'HealthyArea', 'EyesClosed', 'EyesOpen']
# plot_confusion_matrix(test_results, class_names)
