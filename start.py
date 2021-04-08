import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms

writer = SummaryWriter('./log/' + str(np.random.randint(0,10000)))

#data augmentation 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.ColorJitter(hue=.05, saturation=.05),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20)
])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
                         

class SqueezeNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 18, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(18, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sm(self.fc3(x))

        return x

net = Net()

lossFunction = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

for ep in range(200000) :

    imgs = []
    labels = []

    batchsize = 128
    for i in range(batchsize) :
        ind = np.random.randint(0,50000)
        imgs.append(trainset.data[ind])
        labels.append(trainset.targets[ind])

    xt = torch.FloatTensor(imgs).permute(0,3,1,2)
    yt = torch.LongTensor(labels)
    
    vals = net(xt)

    re = torch.argmax(vals, 1)
    acc = ((yt.eq(re.float())).sum()).float()/batchsize

    loss = lossFunction(vals, yt) 

    opt.zero_grad()
    loss.backward()
    opt.step()

    writer.add_scalar('main/loss', loss,ep)
    writer.add_scalar('acc/train', acc,ep)

    #test of the model
    if ep % 100 == 0: 

        imgs = []
        labels = []

        batchsize = 1024
        for i in range(batchsize) :
            ind = np.random.randint(0,10000)
            imgs.append(testset.data[ind])
            labels.append(testset.targets[ind])

        xt = torch.FloatTensor(imgs).permute(0,3,1,2)
        yt = torch.LongTensor(labels)
    
        vals = net(xt)

        re = torch.argmax(vals, 1)
        acc = ((yt.eq(re.float())).sum()).float()/batchsize

        writer.add_scalar('acc/test', acc,ep)

        torch.save(net.state_dict(), './model')


