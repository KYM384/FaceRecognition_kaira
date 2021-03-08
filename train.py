from torchvision import transforms as tf
from torchvision import datasets
from torchvision import utils
from torch.nn import functional as F
from torch import optim
from torch import nn
import torch
import argparse
import numpy as np

from model import get_model, AngularMarginPenalty

def train_loop(net, X, T, arc, criterion, opt):
  y = net(X)
  y = arc(y, T)

  loss = criterion(y,T)
  opt.zero_grad()
  loss.backward()
  opt.step()

  return loss.to("cpu").item()

def train(args):
  assert torch.cuda.is_available(), "No GPU is found"

  device = "cuda"
  num_classes = 10

  transform = tf.Compose([tf.Resize(128),
                          tf.CenterCrop(112),
                          tf.ToTensor(),
                          tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                        ])
  dataset = datasets.ImageFolder(root=args.data, transform=transform)
  train_size = int(len(dataset) * 0.8)
  val_size  = len(dataset)-train_size
  train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch, shuffle=True)
  val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch, shuffle=True)

  print(f"train data : {train_size}  val data : {val_size}")
  
  net = get_model(num_classes)
  net.to(device)

  arc = AngularMarginPenalty(args.margin, args.scale)

  criterion = nn.CrossEntropyLoss()
  criterion.to(device)

  opt = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

  iter = 0
  total_iter = args.epochs * train_size // args.batch
  
  for epoch in range(args.epochs):

    net.train()
    for X,t in train_loader:
      X = X.to(device)
      t = t.to(device)
      loss = train_loop(net, X, t, arc, criterion, opt)

      iter += 1

      if iter == int(total_iter * 0.6):
        opt.lr /= 10
      elif iter == int(total_iter * 0.9):
        opt.lr /= 10

      break

    acc = 0
    net.eval()
    for X,t in val_loader:
      X = X.to(device)
      t = t.to(device)
      y = net(X)
      acc += (y.argmax(1)==t).sum().to("cpu").item()

    print(f"epoch : {epoch+1}  ACC : {acc/val_size:.5f}")

  
  torch.save(net.to("cpu").state_dict(), "arcface.pth")


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Training ArcFace")

  parser.add_argument("--data", type=str, help="path of training images")
  parser.add_argument("--epochs", type=int, help="number of all epochs")
  parser.add_argument("--batch", type=int, help="number of batch size")
  
  parser.add_argument("--margin", default=0.5, type=float, help="value of the angular margin")
  parser.add_argument("--scale", default=64, type=float, help="value of the feature scale")

  args = parser.parse_args()

  train(args)