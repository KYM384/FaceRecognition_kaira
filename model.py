from torch.nn import functional as F
from torch import nn
import torch
from torchvision import models


class LastLayer(nn.Module):
  def __init__(self, in_ch, out_ch, num_classes, noW):
    super().__init__()
    self.W1 = nn.Parameter(torch.randn(out_ch, in_ch))
    self.W2 = nn.Parameter(torch.randn(num_classes, out_ch))
    self.noW = noW
    
  def forward(self,x):
    h = F.linear(x,self.W1)

    h = F.normalize(h)
    if self.noW:
      return h
    
    W = F.normalize(self.W2)
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

