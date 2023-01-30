import torch.nn as nn
import torch.nn.functional as F
import torch
from cc_tmcn.model.common import Conv2d

from cc_tmcn.model.model_uwb import Model_UWB

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d(1, 8, kernel_size=(3), padding=1)
        self.conv2 = Conv2d(8, 8, kernel_size=(5), padding=2)
        self.conv3 = Conv2d(8, 8, kernel_size=(8), padding=4)
        self.conv4 = Conv2d(8, 16, kernel_size=(10), padding=5)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 6, 49))
        self.fc1 = nn.Linear(294, 200)
        self.fc2 = nn.Linear(200, 2)

    def forward_once(self, x):

        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def forward(self, x):
        
        x1 = self.forward_once(x[0])
        x2 = self.forward_once(x[1])

        return x1, x2

class Model_5G(Model_UWB):
    
    def build_model(self):
        """
        Builds the model.

        Returns
        -------
        model : Pytorch object
            Pytorch model

        """

        return Net()