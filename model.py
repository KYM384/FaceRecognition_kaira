from torch.nn import functional as F
from torch import nn
import torch
from torchvision import models


class LastLayer(nn.Module):
  def __init__(self, in_ch, out_ch, num_classes, noW=False):
    super().__init__()
    self.ln = nn.Linear(in_ch, out_ch)
    self.relu = nn.ReLU(inplace=True)

    self.W = nn.Parameter(torch.randn(num_classes, out_ch))
    self.noW = noW
    
  def forward(self,x):
    h = self.relu(self.ln(x))
    h = F.normalize(h)
    if self.noW:
      return h

    W = F.normalize(self.W)
    return F.linear(h, W)


class AngularMarginPenalty(nn.Module):
  def __init__(self, margin, scale):
    super().__init__()
    self.m = margin
    self.s = scale

  def forward(self,x,label):
    one_hot = F.one_hot(label, num_classes=x.shape[1])
    thetas = torch.acos(x)
    y = torch.cos(thetas + one_hot * self.m) * self.s
    return y


def get_model(num_classes, noW=False):
  net = models.resnet50(pretrained=False)
  net.fc = LastLayer(2048, 512, num_classes, noW)
  
  return net
