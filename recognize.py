from torchvision import transforms as tf
from PIL import Image
import numpy as np
import pickle
import torch
import cv2

from model import get_model
from face import Face


class Recognize:
  def __init__(self):
    self.f = Face()
    self.net = get_model(10, True)
    ckpt = torch.load("weights/arcface.pth",map_location=torch.device("cpu"))
    self.net.load_state_dict(ckpt)
    self.net.eval()

    self.transform = tf.Compose([tf.Resize((112, 112)),
                                  tf.ToTensor(),
                                  tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                ])
    self.load()

  def load(self):
    with open("src/names_list.pickle", "rb") as f:
      self.names = pickle.load(f)

    face_vectors = [np.load(f"src/face_vectors/{i}.npy") for i in range(71)]
    self.face_matrix = np.zeros((71, 512))
    for i in range(71):
      self.face_matrix[i] = np.mean(face_vectors[i], axis=0)

  def img2vec(self, img, is_face=False, return_face=False):
    if is_face:
      face = img
    else:
      face = self.f.align_face(img)
      if face is None:
        return None

    face_img = Image.fromarray(np.uint8(face[:, :, ::-1]))
    x = self.transform(face_img).unsqueeze(0)

    with torch.no_grad():
      y = self.net(x)

    vec = y.squeeze(0).detach().numpy()

    if return_face:
      return face, vec

    return vec

  def recognize(self, img, n=5, is_face=False):
    face, vec = self.img2vec(img, is_face, return_face=True)
    scores = self.face_matrix @ vec
    
    ranking = np.argsort(scores)[::-1]

    results = []
    for i in range(n):
      j = ranking[i]
      results.append((self.names[j], scores[j]))

    return face, results