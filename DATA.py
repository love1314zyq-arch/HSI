from dataset import PaviaU
import numpy as np


class DATA(PaviaU):
    def __init__(self, train=True,
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 test_target_transform=None):
        super(DATA, self).__init__(train=train, transform=transform, target_transform=target_transform)
        self.test_transform = test_transform
        self.test_target_transform = test_target_transform
        self.TrainData = []
        self.TestData = []
        self.TrainLabels = []
        self.TestLabels = []

    @staticmethod
    def concatenate(datas, labels):
        con_data = datas[0]
        con_labels = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_labels = np.concatenate((con_labels, labels[i]), axis=0)
        return con_data, con_labels

    def get_train_data(self, classes):
        train_datas, train_labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            train_datas.append(data)
            train_labels.append(np.full((data.shape[0]), label))
        self.TrainData, self.TrainLabels = self.concatenate(train_datas, train_labels)
        print("**********Train Set**********")
        print("the size of train data %s" % (str(self.TrainData.shape)))
        print("the size of train labels %s" % (str(self.TrainLabels.shape)))

    def get_test_data(self, classes):
        test_datas, test_labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            test_datas.append(data)
            test_labels.append(np.full((data.shape[0]), label))
        datas, labels = self.concatenate(test_datas, test_labels)
        self.TestData = datas if self.TestData == [] else np.concatenate((self.TestData, datas), axis=0)
        self.TestLabels = labels if self.TestLabels == [] else np.concatenate((self.TestLabels, labels), axis=0)
        print("**********Test Set**********")
        print("the size of test data %s" % (str(self.TestData.shape)))
        print("the size of test labels %s" % (str(self.TestLabels.shape)))

    def get_test_data_up2now(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas, labels = self.concatenate(datas, labels)
        self.TestData = datas
        self.TestLabels = labels
        print("*" * 15 + "Test Set up to Now" + "*" * 15)
        print("the size of test set up to now is %s" % (str(datas.shape)))
        print("the size of test label up to now is %s" % (str(labels.shape)))

    def __len__(self):
        if self.TrainData != []:
            return len(self.TrainData)
        elif self.TestData != []:
            return len(self.TestData)

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]
