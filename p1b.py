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

# TODO when do I have to set to 0 or 1

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


def compute_test_loss(net, dataloader):
    criterion = nn.BCELoss()

    running_loss = 0
    iter_num = 0
    total_imgs = 0
    for sample_batch in dataloader:
        out = net(Variable(sample_batch['image1'], requires_grad=False).cuda(), Variable(sample_batch['image2'], requires_grad=False).cuda())
        labels = sample_batch['label'].type(torch.FloatTensor)
        labels = labels.view(-1, 1)
        target = Variable(labels, requires_grad=False).cuda()

        loss = criterion(out, target)
        # print 'loss = ', loss.data[0]
        iter_num += 1
        running_loss += loss.data[0]
        net.zero_grad()
        # print running_loss / iter_num
    return running_loss / iter_num

def create_transform_list():
    possible_data_augmenters = [[transforms.RandomHorizontalFlip], [transforms.RandomHorizontalFlip],[transforms.CenterCrop(np.floor(128 * random.uniform(0.7, 1.3))), transforms.Scale((128, 128))], [lambda im: im.rotate(random.randint(-30,30), expand=1 ), transforms.Scale((128, 128))], [lambda im: Image.fromarray(cv2.warpAffine(np.array(im), np.float32([[1, 0, random.randint(-10,10)], [0, 1, random.randint(-10,10)]])))]]
    trans = list()
    trans.append([transforms.Scale((128, 128))])
    num_additional_transformers = random.randint(1,len(possible_data_augmenters))
    indices = np.random.choice(len(possible_data_augmenters), num_additional_transformers, replace=False)
    trans.extend([possible_data_augmenters[i] for i in indices])
    trans.append([transforms.ToTensor()])
    flat = [x for sublist in trans for x in sublist]

    return flat
