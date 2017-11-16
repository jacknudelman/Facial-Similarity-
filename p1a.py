
# TODO when do I have to set to 0 or 1
from data_extraction import *
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader




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


    def forward(self, img1, img2):
        img1 = forward1(self, img1)
        img2 = forward1(self, img2)

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


def forward1(net, x):

    x = net.conv1(x)
    print x.size()
    x = F.relu(x)
    x = net.batchNorm1(x)
    x = net.maxPool(x)

    x = net.conv2(x)
    x = F.relu(x)
    x = net.batchNorm2(x)
    x = net.maxPool(x)

    x = net.conv3(x)
    x = F.relu(x)
    x = net.batchNorm3(x)
    x = net.maxPool(x)

    x = net.conv4(x)
    x = F.relu(x)
    x = net.batchNorm4(x)

    # flatten
    x = x.view(-1, net.num_flat_features(x))

    x = net.linearLayer1(x)

    x = F.relu(x)
    x = net.batchNorm5(x)
    # return x.size()

    return x

def show_batch(sample_batch):
    images_batch1 = sample_batch['image1']
    images_batch2 = sample_batch['image2']

    grid1 = utils.make_grid(images_batch1).numpy().transpose((1, 2, 0))
    plt.imshow(grid1)
    plt.axis('off')
    plt.ioff()
    plt.show()

    grid2 = utils.make_grid(images_batch2).numpy().transpose((1, 2, 0))
    plt.imshow(grid2)
    plt.axis('off')
    plt.ioff()
    plt.show()


net = Net(12).cuda()

transformation = transforms.Compose([transforms.Scale((128, 128)), transforms.ToTensor()])

face_dataset = FaceDataset(csv_file='train.txt', root_dir='lfw/', transformation=transformation)


dataloader = DataLoader(face_dataset, batch_size=net.batchSize, shuffle=True, num_workers=net.batchSize)

learning_rate = 1e-6
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

running_training_loss = 0
training_loss_list = list()
testing_loss_list = list()
xyz_loss = []

iter_num = 0
for epoch in range(15):
    print epoch
# graph stuff
        # set the variable for pltting to 0
    for sample_batch in dataloader:
        out = net(Variable(sample_batch['image1'], requires_grad=True).cuda(), Variable(sample_batch['image2'], requires_grad=True).cuda()).cuda()
        target = sample_batch['label']
        target = np.array([float(i) for i in target])
        target = torch.from_numpy(target).view(net.batchSize, -1)
        target = target.type(torch.FloatTensor)
        target = Variable(target, requires_grad=False).cuda()

        loss = criterion(out, target)
        running_training_loss += loss.data[0]
        net.zero_grad()

        loss.backward()
        optimizer.step()
        xyz_loss.append(loss.data[0])

        iter_num += 1
        if iter_num % 55 == 0:
            training_loss_list.append(running_training_loss)
            testing_loss_list.append(compute_test_loss(net))
            running_training_loss = 0
            mean_loss.append(xyz_loss[-55:].mean())

plt.plot(mean_loss)
plt.plot(training_loss_list)
plt.plot(testing_loss_list)
plt.show()



def compute_test_loss(net) :
    transformation = transforms.Compose([transforms.Scale((128, 128)), transforms.ToTensor()])

    face_dataset = FaceDataset(csv_file='test.txt', root_dir='lfw/', transformation=transformation)

    # global batchSize
    dataloader = DataLoader(face_dataset, batch_size=net.batchSize, shuffle=True, num_workers=net.batchSize)

    criterion = nn.BCELoss()

    running_loss = 0

    for sample_batch in dataloader:
        out = net(Variable(sample_batch['image1'], requires_grad=True).cuda(), Variable(sample_batch['image2'], requires_grad=True).cuda())
        target = sample_batch['label']
        target = np.array([float(i) for i in target])
        target = torch.from_numpy(target).view(net.batchSize, -1)
        target = target.type(torch.FloatTensor)
        target = Variable(target, requires_grad=False)

        loss = criterion(out, target)
        running_loss += loss.data[0]
        net.zero_grad()

    return running_loss



# for i_batch, sample_batch in enumerate(dataloader):
#     print i_batch
#     plt.figure()
#     show_batch(sample_batch)




# input1 = Variable(torch.randn(1, 12, 128, 128), requires_grad=True)
# input2 = Variable(torch.randn(1, 12, 128, 128), requires_grad=True)
#
# out = net(input1, input2)
# loss = net.loss(out, sample['label'])
#
# net.zero_grad()
