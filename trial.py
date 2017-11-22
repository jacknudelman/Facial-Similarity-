import torch
import torchvision
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import os
import cv2
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class lfwDataset(Dataset):

    def __init__(self, csv_file, root_dir, transformation):

        self.root_dir = root_dir
        self.transform = transformation
        self.landmarks_frame = self.read_each_name(csv_file)

    def read_each_name(self, file_name):
        with open(file_name) as f:
            info = open(file_name).read().split()
            all_names = [[None for _ in range(3)] for _ in range(len(info)/3)]
            for x in range(0,len(info)):
                all_names[x/3][x%3] = info[x]
            return all_names

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img1_name = os.path.join(self.root_dir, self.landmarks_frame[idx][0]) # [idx][0]
        img2_name = os.path.join(self.root_dir, self.landmarks_frame[idx][1])

        image1 = Image.open(img1_name)
        image1 = image1.convert('RGB')
        image2 = Image.open(img2_name)
        image2 = image2.convert('RGB')

        label = self.landmarks_frame[idx][2]
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, label


class cnn(nn.Module):

    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, stride = (1,1), padding = 2)
        self.conv2 = nn.Conv2d(64, 128, 5, stride = (1,1), padding = 2)
        self.conv3 = nn.Conv2d(128, 256, 5, stride = (1,1), padding = 2)
        self.conv4 = nn.Conv2d(256, 512, 5, stride = (1,1), padding = 2)

        self.linear1 = nn.Linear(131072, 1024)
        self.linear2 = nn.Linear(2048, 1)

        self.maxPool = nn.MaxPool2d(2, stride = (2,2))
        self.batchSize = 10

        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.batch_norm4 = nn.BatchNorm2d(512)
        self.batch_norm5 = nn.BatchNorm2d(1024)

    def forward(self, image1, image2):
        image1 = forward_each(self, image1)
        image2 = forward_each(self, image2)
        both_results = torch.cat((image1, image2), 1)
        results = F.sigmoid(self.linear2(both_results))

        return results

def forward_each(cnn, x):
    x = cnn.conv1(x)
    x = F.relu(x)
    x = cnn.batch_norm1(x)
    x = cnn.maxPool(x)
    x = cnn.conv2(x)
    x = F.relu(x)
    x = cnn.batch_norm2(x)
    x = cnn.maxPool(x)
    x = cnn.conv3(x)
    x = F.relu(x)
    x = cnn.batch_norm3(x)
    x = cnn.maxPool(x)
    x = cnn.conv4(x)
    x = F.relu(x)
    x = cnn.batch_norm4(x)
    x = x.view((x.data.size())[0], -1)
    x = cnn.linear1(x)
    x = F.relu(x)
    x = cnn.batch_norm5(x)

    return x


def accuracy(label, output):
    result = 0
    counter = 0
    for i in range(0,10):
        counter += 1.0
        value1 = label.data.cpu().numpy()[i]
        value2 = output.data.cpu().numpy()[i]
        if (value1 == value2):
            result += 1
    result = result/counter
    return result


learning_rate = 1e-6
criterion = nn.BCELoss(weight=None, size_average=True)
cnn_model = cnn().cuda()

transform = transforms.Compose([transforms.Scale((128,128)), transforms.ToTensor()])

optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)


def training(cnn_model):
    train_loss = 0
    dataset = lfwDataset(csv_file='train.txt',
                                    root_dir='lfw/', transformation = transform)

    dataloader = DataLoader(dataset, batch_size=10,
                        shuffle=True, num_workers=10)

    train_accuracy = 0
    iterations = 0

    for each in dataloader:
        image1 = Variable(each[0]).cuda()
        image2 = Variable(each[1]).cuda()
        label1 = np.array([float(i) for i in each[2]])
        label1 = torch.from_numpy(label1).view(label1.shape[0], -1)
        label1 = label1.type(torch.FloatTensor)
        label = Variable(label1).cuda()
        output = cnn_model(image1, image2)
        loss = criterion(output, label)

        output = torch.round(output)
        train_accuracy += accuracy(label, output)

        train_loss += loss.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iterations += 1.0
    train_loss = train_loss/iterations

    return train_loss, train_accuracy

def testing(cnn_model):
    dataset = lfwDataset(csv_file='test.txt',
                                    root_dir='lfw/', transformation = transform)
    dataloader = DataLoader(dataset, batch_size=10,
                        shuffle=True, num_workers=10)
    test_loss = 0
    test_accuracy = 0
    iterations = 0

    for each in dataloader:
        image1 = Variable(each[0]).cuda()
        image2 = Variable(each[1]).cuda()
        print type(each[2])
        label1 = np.array([float(i) for i in each[2]])
        label1 = torch.from_numpy(label1).view(label1.shape[0], -1)
        label1 = label1.type(torch.FloatTensor)
        label = Variable(label1).cuda()
        output = cnn_model(image1, image2)
        loss = criterion(output, label)
        output = torch.round(output)
        test_accuracy += accuracy(label, output)
        each_test_accuracy = accuracy(label, output)
        test_loss += loss.data[0]
        iterations += 1.0
    test_loss = test_loss/iterations

    return test_loss, test_accuracy

epochs = 5
all_training_loss = list()
all_testing_loss = list()
all_training_accuracy = list()
all_testing_accuracy = list()

for epoch in range(epochs):
    print epoch
    train_loss, train_accuracy = training(cnn_model)
    train_loss = train_loss
    train_accuracy = train_accuracy/100.0
    print("train loss", train_loss)
    print("train accuracy", train_accuracy)
    all_training_loss.append(train_loss)
    all_training_accuracy.append(train_accuracy)
    test_loss, test_accuracy = testing(cnn_model)
    test_loss = test_loss
    test_accuracy = test_accuracy /100.0
    print("test loss", test_loss)
    print("test accuracy", test_accuracy)
    all_testing_loss.append(test_loss)
    all_testing_accuracy.append(test_accuracy)
    test_loss = 0

print("training loss ", all_training_loss)
print("testing loss", all_testing_loss)
print("training accuracy", all_training_accuracy)
print("testing accuracy", all_testing_accuracy)

torch.save(cnn_model.state_dict(), 'my_weights')

plt.plot(all_training_loss)
plt.plot(all_testing_loss)
plt.plot(all_testing_accuracy)
plt.plot(all_training_accuracy)

plot.show()
