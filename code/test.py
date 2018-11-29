import os
import torch
import torch.nn as nn
import sklearn
from utils import evaluate
from sklearn.metrics import *
from plots import plot_learning_curves
from model import cnn
from loader import custom_data_loader


model = cnn.CNN()
# model.load_state_dict(torch.load("../output/cnn.pth"))
# model.load_state_dict(checkpoint['model_state_dict'])



# torch.save(model, os.path.join(PATH_OUTPUT, 'cnn.pth'))
PATH_TEST_FILE = "../data/test/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = custom_data_loader.XrayLoader(PATH_TEST_FILE)
criterion = nn.CrossEntropyLoss()

criterion.to(device)

best_model = torch.load(os.path.join("../output/", 'cnn.pth'))

_, _, test_results = evaluate(model, device, test_loader, criterion)

roc = 0.0
y1, y2 = zip(*test_results)
try:
    roc = sklearn.metrics.roc_auc_score(y1, y2)
except:
    print("exception")

print("roc " + str(roc))
print("hello")