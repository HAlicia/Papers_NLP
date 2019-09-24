import numpy as np
import random
import pickle
import os


class data_load():
    def __init__(self):
        if not os.path.exists("./data/AG/train_datas"):
            self.train_datas, self.train_labels, self.dev_datas, self.dev_labels, self.test_datas, self.test_labels = self.get_train_dev_test_data()
            char_dict = self.get_char_dict()
            self.train_datas = self.convert_data_to_charid(self.train_datas, char_dict)
            self.dev_datas = self.convert_data_to_charid(self.dev_datas, char_dict)
            self.test_datas = self.convert_data_to_charid(self.test_datas, char_dict)
            self.num_classes = len(np.unique(self.train_labels))
            print(self.train_datas.shape, self.train_labels.shape)
            print(self.dev_datas.shape, self.dev_labels.shape)
            print(self.test_datas.shape, self.test_labels.shape)
            pickle.dump(self.train_datas, open("./data/AG/train_datas", "wb"))
            pickle.dump(self.train_labels, open("./data/AG/train_labels", "wb"))
            pickle.dump(self.dev_datas, open("./data/AG/dev_datas", "wb"))
            pickle.dump(self.dev_labels, open("./data/AG/dev_labels", "wb"))
            pickle.dump(self.test_datas, open("./data/AG/test_datas", "wb"))
            pickle.dump(self.test_labels, open("./data/AG/test_labels", "wb"))
        else:
            self.train_datas = pickle.load(open("./data/AG/train_datas", "rb"))
            self.train_labels = pickle.load(open("./data/AG/train_labels", "rb"))
            self.dev_datas = pickle.load(open("./data/AG/dev_datas", "rb"))
            self.dev_labels = pickle.load(open("./data/AG/dev_labels", "rb"))
            self.test_datas = pickle.load(open("./data/AG/test_datas", "rb"))
            self.test_labels = pickle.load(open("./data/AG/test_labels", "rb"))
            self.num_classes = len(np.unique(self.train_labels))
            print(self.train_datas.shape, self.train_labels.shape)
            print(self.dev_datas.shape, self.dev_labels.shape)
            print(self.test_datas.shape, self.test_labels.shape)

    def read_file(self, path):
        datas = open(path, "r", encoding="utf-8").read().splitlines()
        labels = []
        texts = []
        for i, data in enumerate(datas):
            data = data.lower()
            if i % 10000 == 0:
                print(i, len(datas))
            data = data.split(',"', 1)
            labels.append(int(data[0].strip("\"")) - 1)
            texts.append(data[1].lower())
        return texts, labels

    def get_train_dev_test_data(self):
        datas, labels = self.read_file("./data/AG/train.csv")
        n = int(len(datas) * 0.8)
        cc = list(zip(datas, labels))
        random.shuffle(cc)
        datas[:], labels[:] = zip(*cc)
        train_datas = datas[0:n]
        train_labels = labels[0:n]
        train_labels = np.array(train_labels)
        dev_datas = datas[n:]
        dev_labels = labels[n:]
        dev_labels = np.array(dev_labels)
        test_datas, test_labels = self.read_file("./data/AG/test.csv")
        test_labels = np.array(test_labels)
        return train_datas, train_labels, dev_datas, dev_labels, test_datas, test_labels

    def get_char_dict(self):
        chars = '''abcdefghijklmnopqrstuvwxyz
        0123456789
        -,;.!?:'"/\|_@#$%ˆ&*˜‘
        +-=<>()[]{} '''
        char_dict = {"<pad>": 0, "<unk>": 1}
        for char in chars:
            char_dict[char] = len(char_dict)
        return char_dict

    def convert_data_to_charid(self, datas, char_dict):
        new_datas = []
        for i, data in enumerate(datas):
            if i % 10000 == 0:
                print(i, len(datas))
            new_datas.append([])
            for j, char in enumerate(data):
                if j == 1014:
                    break
                new_datas[i].append(char_dict.get(char, 1))
            new_datas[i] = new_datas[i] + [0] * (1014 - len(new_datas[i]))
        new_datas = np.array(new_datas)
        return new_datas


if __name__ == "__main__":
    data_load = data_load()
