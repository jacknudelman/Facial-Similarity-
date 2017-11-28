import matplotlib
matplotlib.use('Agg')
from data_extraction import *
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import sys
import random


class Net(nn.Module):

    def __init__(self, batchSize):
        super(Net, self).__init__()
        self.batchSize = batchSize
        self.maxPool = nn.MaxPool2d(2, stride=(2,2))
        self.linearLayer1 = nn.Linear(131072, 1024)
        self.linearLayer2 = nn.Linear(2048, 1)

        self.conv1 = nn.Conv2d(3, 64, 5, stride=(1,1), padding=2)
        self.batchNorm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 5, stride=(1,1), padding=2)
        self.batchNorm2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=(1,1), padding=1)
        self.batchNorm3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, 3, stride=(1,1), padding=1)
        self.batchNorm4 = nn.BatchNorm2d(512)

        self.batchNorm5 = nn.BatchNorm2d(1024)

    def forward1(self, x):

        x = self.conv1(x)
        # x.size()
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.maxPool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.maxPool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.maxPool(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)

        # flatten
        x = x.view(-1, self.num_flat_features(x))

        x = self.linearLayer1(x)

        x = F.relu(x)
        x = self.batchNorm5(x)
        # print x.size()
        return x

    def forward(self, img1, img2):
        img1 = self.forward1(img1)
        img2 = self.forward1(img2)

        z = torch.cat((img1, img2), 1)
        z = self.linearLayer2(z)
        z = F.sigmoid(z)

        return z

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def play(weight_path):
    net = Net(25).cuda()
    net.train()

    train_transformation = transforms.Compose([transforms.Scale((128, 128)), transforms.ToTensor()])
    train_face_dataset = FaceDataset(csv_file='train.txt', root_dir='lfw/', transform=train_transformation)
    train_dataloader = DataLoader(train_face_dataset, batch_size=net.batchSize, shuffle=True, num_workers=net.batchSize)

    test_transformation = transforms.Compose([transforms.Scale((128, 128)), transforms.ToTensor()])
    test_face_dataset = FaceDataset(csv_file='test.txt', root_dir='lfw/', transform=test_transformation)
    test_dataloader = DataLoader(test_face_dataset, batch_size=net.batchSize, shuffle=True, num_workers=net.batchSize)
    # print 'got datasets'

    criterion = nn.BCELoss()

    learning_rate = 5e-6
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    training_loss_list = list()
    testing_loss_list = list()
    average_testing_loss = list()

    iter_num = 0
    running_training_loss = 0
    num_correctly_matched = 0
    total_num_correctly_matched = 0
    total_num_imgs = 0
    test_total_num_correctly_matched = 0
    test_total_num_imgs = 0
    file_name = 'fig'
    if ('--save' in sys.argv):
        print 'augmenting'
        file_name = 'aug_fig'
        train_face_dataset = RandFaceDataset(csv_file='train.txt', root_dir='lfw/', transform=test_transformation)
        train_dataloader = DataLoader(train_face_dataset, batch_size=net.batchSize, shuffle=True, num_workers=net.batchSize)
    for epoch in range(25):
        print epoch
        for sample_batch in train_dataloader:
            iter_num += 1
            img1 = Variable(sample_batch[0], requires_grad=True).type(torch.FloatTensor).cuda()
            img2 = Variable(sample_batch[1], requires_grad=True).type(torch.FloatTensor).cuda()
            labels = torch.from_numpy(np.array([float(i) for i in sample_batch[2]])).view(-1, 1)
            labels = labels.type(torch.FloatTensor)
            target = Variable(labels).cuda()

            loss = train_bce(net, optimizer, img1, img2, target, criterion)

            training_loss_list.append(loss)


        out_acc = test_bce(test_dataloader, net)
        print 'testing accuracy', out_acc[0]
        testing_loss_list.append(out_acc[0])

        if (epoch + 1) % 5 == 0:
            out_acc = test_bce(train_dataloader, net)
            print 'train accuracy', out_acc[0]


    # torch.save(net.state_dict(), weight_path)

    print len(training_loss_list)
    print len(testing_loss_list)

    x_training = np.linspace(0, iter_num, len(training_loss_list))
    plt.plot(x_training, training_loss_list)

    plt.title('training loss')
    plt.savefig(file_name)
    plt.clf()

    x_raw_testing = np.linspace(0, iter_num, len(testing_loss_list))
    plt.plot(x_raw_testing, testing_loss_list)

    plt.title('testing accuracy')
    plt.savefig(file_name + '_accuracy')


def train_bce(net, optimizer, img1, img2, target, criterion):
    # net.train()

    optimizer.zero_grad()
    out = net(img1, img2)
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()
    return loss.data[0]

def test_bce(loader, net):
    # net.eval()

    num_correct = 0
    num_images = 0

    for sample_batch in loader:
        img1 = Variable(sample_batch[0], requires_grad=True).type(torch.FloatTensor).cuda()
        img2 = Variable(sample_batch[1], requires_grad=True).type(torch.FloatTensor).cuda()
        labels = torch.from_numpy(np.array([float(i) for i in sample_batch[2]])).view(-1, 1)
        labels = labels.type(torch.FloatTensor)
        target = Variable(labels).cuda()

        out = net(img1, img2)
        temp = torch.round(out)

        for i in range(temp.size()[0]):
            if (temp.data[i][0] == target.data[i][0]):
                num_correct += 1
        num_images += temp.size()[0]
    return [float(num_correct)/float(num_images), num_correct, num_images]


if '--save' in sys.argv:
    weight_path_index = sys.argv.index('--save') + 1
    weight_path = sys.argv[weight_path_index]
    play(weight_path)
    # train(weight_path)
if '--load' in sys.argv:
    weight_path_index = sys.argv.index('--load') + 1
    weight_path = sys.argv[weight_path_index]
    net = Net(25).cuda()
    # net.eval()
    net.load_state_dict(torch.load(weight_path))
    # net.eval()
    # print 'created net'
    train_transformation = transforms.Compose([transforms.Scale((128, 128)), transforms.ToTensor()])
    train_face_dataset = FaceDataset(csv_file='train.txt', root_dir='lfw/', transform=train_transformation)
    train_dataloader = DataLoader(train_face_dataset, batch_size=net.batchSize, shuffle=True, num_workers=net.batchSize)

    test_transformation = transforms.Compose([transforms.Scale((128, 128)), transforms.ToTensor()])
    test_face_dataset = FaceDataset(csv_file='test.txt', root_dir='lfw/', transform=test_transformation)
    test_dataloader = DataLoader(test_face_dataset, batch_size=net.batchSize, shuffle=True, num_workers=net.batchSize)

    acc[0] = test_bce(train_dataloader, net)
    print 'training acuracy = ', acc[0]

    acc = test_bce(test_dataloader, net)
    print 'testing accuracy = ', acc[0]
