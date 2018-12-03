import os
import numpy as np
import preprocess.salt_pepper_transform as spt
import preprocess.reflection_transformation as refl
import preprocess.rotation_transform as rotation
import random

class XrayLoader():
    def __init__(self, path = "", dataset = "original", augmentation = "additive_augmentation", negative_batch_size = 15, positive_batch_size = 1):
        self.path = path
        self.dataset = dataset
        self.augmentation = augmentation
        self.negative_batch_size = negative_batch_size
        self.positive_batch_size = positive_batch_size
        self.positive_directory_list = os.listdir(path + "positive/")
        self.negative_directory_list = os.listdir(path + "negative/")
        self.positive_size = len(self.positive_directory_list)
        self.negative_size = len(self.negative_directory_list)
        self.positive_file_index = 1
        self.negative_file_index = 1
        self.positive_data_index = 0
        self.negative_data_index = 0
        self.positive_data = np.load(self.path + "positive/" + self.positive_directory_list[0])
        self.negative_data = np.load(self.path + "negative/" + self.negative_directory_list[0])

    def reset(self):
        self.positive_directory_list = os.listdir(self.path + "positive/")
        self.negative_directory_list = os.listdir(self.path + "negative/")
        self.positive_file_index = 1
        self.negative_file_index = 1
        self.positive_data_index = 0
        self.negative_data_index = 0
        self.positive_data = np.load(self.path + "positive/" + self.positive_directory_list[0])
        self.negative_data = np.load(self.path + "negative/" + self.negative_directory_list[0])




    def get_next_negative(self):
        if self.negative_data is None:
            return None
        l = len(self.negative_data)
        i = self.negative_data_index
        self.negative_data_index += self.negative_batch_size
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
        self.positive_data_index += self.positive_batch_size
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
        if p is not None and len(p) == 0:
            p = self.get_next_positive()
        n = self.get_next_negative()
        if n is not None and len(n) == 0:
            n = self.get_next_negative()
        if self.dataset != "original" and "train" not in self.dataset:
            p = self.apply_augmentation(p)
            n = self.apply_augmentation(n)



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
        elif n is not None and "train" not in self.path:
            data = n
            n_size = len(n)
            n_labels = np.ones(n_size)
            labels = n_labels
        else:
            data = None
            labels = None

        data = self.expand(data)
        return (data, labels)

    def apply_augmentation(self, data):
        if data is None:
            return None

        new_data = []

        if self.dataset == "salt_and_pepper":
            if self.augmentation == "additive_augmentation":
                for i in range(len(data)):
                    new_data.append(data[i])
                    new_data.append(spt.salt_and_pepper(data[i]))
            else:
                for i in range(len(data)):
                    rand = random.randint(1, 100)
                    if rand >= 0 and rand <= 60:
                        new_data.append(data[i])
                    else:
                        new_data.append(spt.salt_and_pepper(data[i]))
        if self.dataset == "reflection":
            if self.augmentation == "additive_augmentation":
                for i in range(len(data)):
                    new_data.append(data[i])
                    new_data.append(refl.reflection(data[i]))
            else:
                for i in range(len(data)):
                    rand = random.randint(1, 100)
                    if rand >= 0 and rand <= 60:
                        new_data.append(data[i])
                    else:
                        new_data.append(refl.reflection(data[i]))

        if self.dataset == "rotation":
            if self.augmentation == "additive_augmentation":
                for i in range(len(data)):
                    new_data.append(data[i])
                    new_data.append(rotation.rotation(data[i]))
            else:
                for i in range(len(data)):
                    rand = random.randint(1, 100)
                    if rand >= 0 and rand <= 60:
                        new_data.append(data[i])
                    else:
                        new_data.append(rotation.rotation(data[i]))

        if self.dataset == "all":
            if self.augmentation == "additive_augmentation":
                for i in range(len(data)):
                    new_data.append(data[i])
                    new_data.append(rotation.rotation(data[i]))
                    new_data.append(rotation.rotation(data[i]))
                    new_data.append(rotation.rotation(data[i]))
                    new_data.append(rotation.rotation(data[i]))
            else:
                for i in range(len(data)):
                    rand = random.randint(1, 100)
                    if rand >= 0 and rand <= 25:
                        new_data.append(data[i])
                    if rand >= 0 and rand <= 50:
                        new_data.append(spt.salt_and_pepper(data[i]))
                    if rand >= 0 and rand <= 75:
                        new_data.append(rotation.rotation(data[i]))
                    else:
                        new_data.append(refl.reflection(data[i]))

        return np.array(new_data)

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


# loader = XrayLoader("../../data/train/")
#
# batch, labels = loader.get_next_batch()
#
# while batch is not None:
#     batch, labels = loader.get_next_batch()
#     print(str(len(batch)))
# print("hey")
# print("done")