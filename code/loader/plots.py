import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(original, s_and_p):
	plt.plot(original)
	plt.plot(s_and_p)
	# plt.plot(valid_losses)
	plt.ylabel("AUC_ROC")
	plt.xlabel("Epoch")
	plt.title("ROC Curve")
	plt.legend(['original', 's_and_p'])
	plt.show()


	pass


def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	actual, predicted = list(zip(*results))
	cnf_matrix = confusion_matrix(actual, predicted)
	cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
	plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)

	plt.colorbar()
	tick_marks = np.arange(len(class_names))
	plt.xticks(tick_marks, class_names, rotation=45)
	plt.yticks(tick_marks, class_names)

	fmt = '.2f'
	thresh = cnf_matrix.max() / 2.
	for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
		plt.text(j, i, format(cnf_matrix[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cnf_matrix[i, j] > thresh else "black")

	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.title("Normalized Confusion Matrix")
	plt.tight_layout()

	plt.show()
	pass
