import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):

    def __init__(
        self,
        in_chan,
        out_chan,
        **kwargs):

        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, **kwargs)
        self.bn = nn.BatchNorm2d(out_chan, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)
