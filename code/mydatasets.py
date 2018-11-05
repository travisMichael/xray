import torch
from torch.utils.data import TensorDataset, Dataset
import pandas as pd
import numpy as np

def load_seizure_dataset(path, model_type):
	df = pd.read_csv(path)
#	d = np.load("aaa.npy")
	# data = torch.zeros((2, 2))
	iData = df.iloc[:, 0:178].values
	data = torch.tensor(iData, dtype=torch.float32)  # .unsqueeze(dim=1)
	# target = torch.zeros(2)
	iTarget = df.iloc[:, 178].values
	target = torch.tensor(iTarget)
	target = target - 1
	target = target  # .unsqueeze(dim=1)
	dataset = TensorDataset(data, target)

	return dataset




