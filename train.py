from torchvision import transforms as tf
from torchvision import datasets
from torchvision import utils
from torch.nn import functional as F
from torch import optim
from torch import nn
import torch
import numpy as np
import argparse
import os

from model import get_model, AngularMarginPenalty

def train_loop(net, X, T, arc, criterion, opt, scheduler):
  y = net(X)
  y = arc(y, T)

  loss = criterion(y,T)
  opt.zero_grad()
  loss.backward()
  opt.step()
  scheduler.step()

  return loss.to("cpu").item()


def test(net, data_loader, device):
  acc = 0
  net.eval()
  for X,t in data_loader:
    X = X.to(device)
    t = t.to(device)

    with torch.no_grad():
      y = net(X)
      acc += (y.argmax(1)==t).sum().to("cpu").item()

  return acc / len(data_loader.dataset)


def train(args):
  assert torch.cuda.is_available(), "No GPU is found"

  np.random.seed(0)
  torch.manual_seed(0)
  torch.cuda.manual_seed(0)

  device = "cuda"
  num_classes = 10
  total_epoch = args.epoch

  transform_train = tf.Compose([tf.Resize(128),
                                tf.CenterCrop(112),
                                tf.RandomHorizontalFlip(),
                                tf.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                tf.ToTensor(),
                                tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                              ])
  transform_val = tf.Compose([tf.Resize(128),
                                tf.CenterCrop(112),
                                tf.ToTensor(),
                                tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                              ])

  base_path = args.data
  train_data = datasets.ImageFolder(root=os.path.join(base_path, "train"), transform=transform_train)
  val_data = datasets.ImageFolder(root=os.path.join(base_path, "val"), transform=transform_val)
  test_data = datasets.ImageFolder(root=os.path.join(base_path, "test"), transform=transform_val)

  train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch, shuffle=True)
  val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch, shuffle=False)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch, shuffle=False)

  print(f"train data : {len(train_data)}  val data : {len(val_data)}  test data : {len(test_data)}")
  
  net = get_model(num_classes)
  net.to(device)

  arc = AngularMarginPenalty(args.margin, args.scale)

  criterion = nn.CrossEntropyLoss()
  criterion.to(device)

  opt = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
  scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[int(total_epoch*0.6), int(total_epoch*0.9)], gamma=0.1)

  min_acc = 1
  
  for epoch in range(total_epoch):

    net.train()
    for X,t in train_loader:
      X = X.to(device)
      t = t.to(device)
      loss = train_loop(net, X, t, arc, criterion, opt, scheduler)

    acc = test(net, val_loader, device)

    print(f"epoch : {epoch+1}  ACC : {acc:.5f}")
    if acc < min_acc:
      min_acc = acc
      torch.save(net.state_dict(), "weights/arcface.pth")


  net.load_state_dict(torch.load("weights/arcface.pth"))
  acc = test(net, test_loader, device)
  print(f"Test  ACC : {acc:.5f}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Training ArcFace")

  parser.add_argument("--data",               type=str, help="path of training images")
  parser.add_argument("--epoch", default=100, type=int, help="number of all epochs")
  parser.add_argument("--batch", default=256, type=int, help="number of batch size")
  
  parser.add_argument("--margin", default=0.5, type=float, help="value of the angular margin")
  parser.add_argument("--scale",  default=32,  type=float, help="value of the feature scale")

  args = parser.parse_args()

  train(args)