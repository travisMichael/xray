import os
import numpy as np

class XrayLoader():
    def __init__(self, path = "", batch_size = 16):
        self.path = path
        self.batch_size = batch_size / 2
        self.positive_directory_list = os.listdir(path + "positive/")
        self.negative_directory_list = os.listdir(path + "negative/")
        self.positive_size = len(self.positive_directory_list)
        self.negative_size = len(self.negative_directory_list)
        self.positive_data = None
        self.negative_data = None
        self.positive_file_index = 0
        self.negative_file_index = 0
        self.positive_data_index = 0
        self.negative_data_index = 0
        self.initialize()

    def initialize(self):
        self.positive_data = np.load(self.path + "positive/" + self.positive_directory_list[self.positive_file_index])
        self.negative_data = np.load(self.path + "negative/" + self.negative_directory_list[self.negative_file_index])
        self.positive_file_index += 1
        self.negative_file_index += 1

    def has_next_batch(self):
        return True

    def get_next_negative(self):
        if self.negative_data is None:
            return None
        l = len(self.negative_data)
        i = self.negative_data_index
        self.negative_data_index += 8
        data = self.negative_data[i:self.negative_data_index]

        if l < self.negative_data_index:
            if self.negative_file_index >= len(self.negative_directory_list):
                self.negative_data = None
            else:
                print("switching negative")
                self.negative_data = np.load(self.path + "negative/" + self.negative_directory_list[self.negative_file_index])
                self.negative_data_index = 0
                self.negative_file_index += 1

        return data

    def get_next_positive(self):
        if self.positive_data is None:
            return None
        l = len(self.positive_data)
        i = self.positive_data_index
        self.positive_data_index += 8
        data = self.positive_data[i:self.positive_data_index]

        if l < self.positive_data_index:
            if self.positive_file_index >= len(self.positive_directory_list):
                self.positive_data = None
            else:
                print("switching positive")
                self.positive_data = np.load(self.path + "positive/" + self.positive_directory_list[self.positive_file_index])
                self.positive_data_index = 0
                self.positive_file_index += 1

        return data

    def get_next_batch(self):

        p = self.get_next_positive()
        n = self.get_next_negative()


        if n is not None and p is not None:
            p_size = len(p)
            n_size = len(n)

            n_labels = np.zeros(n_size)
            p_labels = np.ones(p_size)
            data = np.concatenate((n, p))
            labels = np.hstack((n_labels, p_labels))
        elif p is not None:
            data = p
            p_size = len(p)
            p_labels = np.ones(p_size)
            labels = p_labels
        elif n is not None:
            data = n
            n_size = len(n)
            n_labels = np.ones(n_size)
            labels = n_labels
        else:
            data = None
            labels = None

        data = self.expand(data)
        return (data, labels)

    def expand(self, data):
        if data is None:
            return None
        new_data = []
        for i in range(len(data)):
            channels = []
            channels.append(data[i])
            channels.append(data[i])
            channels.append(data[i])
            new_data.append(channels)

        return np.array(new_data)


    def reset(self):
        self.index = 0

    def iterate(self, path):
        batch_size = 16
        positive_directory_list = os.listdir(path + "positive/")
        negative_directory_list = os.listdir(path + "negative/")
        for i in negative_directory_list:
            print(i)
        print("here")


# loader = XrayLoader("../../data/train/")
#
# batch, labels = loader.get_next_batch()
#
# while batch is not None:
#     batch, labels = loader.get_next_batch()
#     print(str(len(batch)))
# print("hey")
# print("done")