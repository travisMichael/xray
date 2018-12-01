import os
import time
import numpy as np
import torch
from skimage import io, transform
import sklearn


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def max_num_in_list(list):
	largest = 0
	for item in list:
		if 'original' in item:
			current = int(item.split('orig')[0])
			if current > largest:
				largest = current
	return largest

def calculate_weigths(path):
	n_list = os.listdir(path + "/data/train/negative/")
	p_list = os.listdir(path + "/data/train/positive/")
	n_size = max_num_in_list(n_list)
	p_size = max_num_in_list(p_list)

	total = n_size + p_size
	weights = [(float(n_size) / total), (float(p_size) / total)]
	return weights

def compute_batch_accuracy(output, target):
	"""Computes the accuracy for a batch"""
	with torch.no_grad():

		batch_size = target.size(0)
		_, pred = output.max(1)
		correct = pred.eq(target).sum()

		return correct * 100.0 / batch_size


def train(model, device, data_loader, criterion, optimizer, epoch, print_freq=10):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	accuracy = AverageMeter()

	model.train()

	end = time.time()
	index = 1
	i = 0
	total_load = 0
	total_train = 0
	while True:


		start_load_time = time.time()
		input, target = data_loader.get_next_batch()

		if input is None:
			data_loader.reset()
			break


		input = torch.tensor(input, dtype=torch.float)
		target = torch.tensor(target, dtype=torch.long)
		end_load_time = time.time()
		total_load +=  (end_load_time - start_load_time)


		start_train_time = time.time()
		if isinstance(input, tuple):
			input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
		else:
			input = input.to(device)
		target = target.to(device)

		optimizer.zero_grad()
		output = model(input)
		# target = target.squeeze(dim=1)
		loss = criterion(output, target)
		assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		losses.update(loss.item(), target.size(0))
		accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))

		end_train_time = time.time()
		total_train +=  (end_train_time - start_train_time)
		index += 1
		if i % print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
				epoch, i, 100, batch_time=batch_time, 					# data_loader.size
				data_time=data_time, loss=losses, acc=accuracy))
		i += 1

	return losses.avg, accuracy.avg


def evaluate(model, device, data_loader, criterion, print_freq=10):
	batch_time = AverageMeter()
	losses = AverageMeter()
	accuracy = AverageMeter()

	results = []

	model.eval()
	i = 0

	with torch.no_grad():
		end = time.time()
		while True:

			input, target = data_loader.get_next_batch()

			if input is None:
				data_loader.reset()
				break

			input = torch.tensor(input, dtype=torch.float)
			target = torch.tensor(target, dtype=torch.long)

			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			target = target.to(device)

			output = model(input)
			# target = target.squeeze(dim=1)
			loss = criterion(output, target)

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			losses.update(loss.item(), target.size(0))
			accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))

			y_true = target.detach().to('cpu').numpy().tolist()
			y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
			results.extend(list(zip(y_true, y_pred)))
			# y1, y2 = zip(*results)
			# f1 = sklearn.metrics.f1_score(y1, y2)
			# roc = sklearn.metrics.roc_auc_score(y2, y1)
			i += 1

			if i % print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
					i, 100, batch_time=batch_time, loss=losses, acc=accuracy))

	return losses.avg, accuracy.avg, results


