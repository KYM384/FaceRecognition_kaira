from imutils import face_utils
import numpy as np
import dlib
import cv2

class Face:
  def __init__(self):
    self.det = dlib.get_frontal_face_detector()
    self.prd = dlib.shape_predictor("weights/shape_predictor_68_face_landmarks.dat")

  def face_detect(self,img):
    gry = img[:,:,1:2]
    faces = self.det(gry,1)
    lms = np.zeros((len(faces),4,2))
    for i,face in enumerate(faces):
      lm = self.prd(gry,face)
      lm = face_utils.shape_to_np(lm)
      pts = [range(36,42),range(42,48),range(30,36),range(48,68)]
      lms[i,0] = np.mean(lm[36:42],axis=0)
      lms[i,1] = np.mean(lm[42:48],axis=0)
      lms[i,2] = np.mean(lm[48:68],axis=0)
      lms[i,3] = np.mean(lm[30:36],axis=0)
    return lms

  def face_rotate(self,img,lms):
    H,W,_ = img.shape
    img = np.pad(img,[[H//2,H//2],[W//2,W//2],[0,0]],mode="symmetric")
    k_size = max(W,H)//20
    k_size += 1-k_size%2
    blur = cv2.blur(img,(k_size,k_size))

    blur[H//2:H//2+H,W//2:W//2+W] = img[H//2:H//2+H,W//2:W//2+W]
    lms += np.array([[W//2,H//2]])

    theta = np.arctan((lms[:,0,1]-lms[:,1,1]) / (lms[:,0,0]-lms[:,1,0]))
    theta = theta*180/np.pi
    center = np.mean(lms,axis=1,keepdims=True)
    radius = np.max(np.abs(lms-center),axis=(1,2))

    faces = []

    for t,c,r in zip(theta,center[:,0],radius):
      rotate_matrix = cv2.getRotationMatrix2D(tuple(c),t,1.0)
      rotated_img = cv2.warpAffine(blur,rotate_matrix,(2*W,2*H))

      cx,cy = c
      cy -= r//2

      x0 = max(int(cx-r*2.8),0)
      y0 = max(int(cy-r*2.8),0)
      x1 = min(int(cx+r*2.8),2*W)
      y1 = min(int(cy+r*2.8),2*H)

      face = rotated_img[y0:y1,x0:x1]
      faces.append(face)

    return faces

  def align_face(self,img):
    lms = self.face_detect(img)
    face = self.face_rotate(img,lms)
    if len(face) == 0:
      return None

    return face[0]
