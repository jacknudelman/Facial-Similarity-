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


def compute_test_loss(net, dataloader):
    criterion = nn.BCELoss()

    running_loss = 0
    iter_num = 0
    num_images = 0
    num_correctly_matched = 0
    for sample_batch in dataloader:
        out = net(Variable(sample_batch[0], requires_grad=False).cuda(), Variable(sample_batch[1], requires_grad=False).cuda())
        labels = torch.from_numpy(np.array([float(i) for i in sample_batch[2]])).view(-1, 1)
        labels = labels.type(torch.FloatTensor)
        target = Variable(labels).cuda()

        loss = criterion(out, target)
        for i in range(target.size()[0]):
            if((target.data[i][0] == 1 and out.data[i][0] >= 0.5) or (target.data[i][0] == 0 and out.data[i][0] < 0.5)):
                num_correctly_matched += 1
        iter_num += 1
        num_images += target.size()[0]
        running_loss += loss.data[0]
        net.zero_grad()
    return [(running_loss / iter_num), num_correctly_matched, num_images]

def create_transform_list():
    # , [lambda im: im.rotate(random.randint(-30,30), expand=1 ), transforms.Scale((128, 128))]
    possible_data_augmenters = [[transforms.RandomHorizontalFlip()], [lambda im: im.rotate(random.randint(-30,30), expand=1 ), transforms.Scale((128, 128))], [transforms.RandomHorizontalFlip()],[transforms.CenterCrop(np.floor(128 * random.uniform(0.7, 1.3))), transforms.Scale((128, 128))], [lambda im: Image.fromarray(cv2.warpAffine(np.array(im), np.float32([[1, 0, random.randint(-10,10)], [0, 1, random.randint(-10,10)]]), (np.array(im).shape[1], np.array(im).shape[0])))]]
    trans = list()
    trans.append([transforms.Scale((128, 128))])
    num_additional_transformers = random.randint(1,len(possible_data_augmenters))
    indices = np.random.choice(len(possible_data_augmenters), num_additional_transformers, replace=False)
    print indices
    trans.extend([possible_data_augmenters[i] for i in indices])
    trans.append([transforms.ToTensor()])
    # print 'augmented'
    flat = [x for sublist in trans for x in sublist]

    return flat

net = Net(40).cuda()
# print 'created net'
train_transformation = transforms.Compose([transforms.Scale((128, 128)), transforms.ToTensor()])
train_face_dataset = FaceDataset(csv_file='train.txt', root_dir='lfw/', transform=train_transformation)
train_dataloader = DataLoader(train_face_dataset, batch_size=net.batchSize, shuffle=True, num_workers=net.batchSize)

test_transformation = transforms.Compose([transforms.Scale((128, 128)), transforms.ToTensor()])
test_face_dataset = FaceDataset(csv_file='test.txt', root_dir='lfw/', transform=test_transformation)
test_dataloader = DataLoader(test_face_dataset, batch_size=net.batchSize, shuffle=True, num_workers=net.batchSize)
# print 'got datasets'
learning_rate = 1e-6
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

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
if ('--augment' in sys.argv):
    print 'augmenting'
    file_name = 'aug_fig'
    train_face_dataset = RandFaceDataset(csv_file='test.txt', root_dir='lfw/', transform=test_transformation)
    train_dataloader = DataLoader(train_face_dataset, batch_size=net.batchSize, shuffle=True, num_workers=net.batchSize)
for epoch in range(10):
    print epoch
    num_images = 0

    for sample_batch in train_dataloader:
        # if '--augment' in sys.argv:
        #     if random.uniform(0.0, 1.0) > 0.3:
        #         # print 'getting transforms'
        #         train_face_dataset.transform = transforms.Compose(create_transform_list())
        # print sample_batch[0].size()
        out = net(Variable(sample_batch[0]).cuda(), Variable(sample_batch[1]).cuda())
        print 'got out'
        labels = torch.from_numpy(np.array([float(i) for i in sample_batch[2]])).view(-1, 1)
        labels = labels.type(torch.FloatTensor)
        target = Variable(labels).cuda()
        for i in range(target.size()[0]):
            if((target.data[i][0] == 1 and out.data[i][0] >= 0.5) or (target.data[i][0] == 0 and out.data[i][0] < 0.5)):
                num_correctly_matched += 1
        num_images += target.size()[0]

        loss = criterion(out, target)
        running_training_loss += loss.data[0]
        net.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss_list.append(loss.data[0])
        iter_num += 1
        if iter_num % 11 == 0:
            training_loss_list.append(running_training_loss / 11)
            running_training_loss = 0
            [testloss, test_num_correct, test_tested] = compute_test_loss(net, test_dataloader)
            testing_loss_list.append(testloss)
            test_total_num_correctly_matched += test_num_correct
            test_total_num_imgs += test_tested

            # if iter_num % 9 == 0:
            #     print 'iter_num = ', iter_num
            #     av = np.average(testing_loss_list[-10:])
            #     print av
            #     average_testing_loss.append(av)

    # print num_images
    print 'train accuracy on epoch ', epoch,  ' is ', float(num_correctly_matched)/ num_images
    print 'current average testing accuracy is ', float(test_total_num_correctly_matched)/ float(test_total_num_imgs)
    total_num_correctly_matched += num_correctly_matched
    total_num_imgs += num_images
    num_correctly_matched = 0
#
print 'total train correct = ', total_num_correctly_matched
print 'total train  = ', total_num_imgs
print 'total test correct = ', test_total_num_correctly_matched
print 'total test  = ', test_total_num_imgs
print 'average train accuracy is ', float(total_num_correctly_matched)/ float(total_num_imgs)
print 'average test accuracy is ', float(test_total_num_correctly_matched)/ float(test_total_num_imgs)
torch.save(net, 'net_state')

print len(training_loss_list)
print len(testing_loss_list)
print len(average_testing_loss)

x_training = np.linspace(0, iter_num, len(training_loss_list))
plt.plot(x_training, training_loss_list)

x_raw_testing = np.linspace(0, iter_num, len(testing_loss_list))
plt.plot(x_raw_testing, testing_loss_list)

plt.savefig(file_name)
plt.title('losses')


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
