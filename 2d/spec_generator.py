import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
import imageio

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 2, 8)
        self.conv2 = nn.Conv2d(2, 1, 8)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 8, 4)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 4, 4)
        return x

class SpecGenerator:
    def __init__(self):
        pass

if __name__ == '__main__':
    net = Net()
    img0 = imageio.imread('data/drawings/0.png')
    img1 = imageio.imread('data/drawings/1.png')
    img = np.stack((img0, img1), 2)
    img = img.reshape((1, 2, 256, 256))
    tensor_img = torch.from_numpy(img).type(torch.FloatTensor)
    tensor_rand = torch.randn(1, 2, 256, 256)
    print(tensor_img.type())
    print(tensor_rand.type())
    result = net(tensor_img)
    print(result)

