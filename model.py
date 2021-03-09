from torch.nn import functional as F
from torch import nn
import torch
from torchvision import models


class LastLayer(nn.Module):
  def __init__(self, in_ch, num_classes):
    super().__init__()
    self.W = nn.Parameter(torch.randn(num_classes, in_ch))
    
  def forward(self,x):
    x = F.normalize(x)
    W = F.normalize(self.W)
    return F.linear(x, W)


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


def get_model(num_classes):
  net = models.resnet18(pretrained=False)
  net.fc = LastLayer(512, num_classes)
  return net

