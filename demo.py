from recognize import Recognize
import argparse
import cv2
import os

def main(args):
  rec = Recognize()
  img = cv2.imread(args.image)

  assert img, f"No such file: '{args.image}'"

  results = rec.recognize(img)
  assert results, "No face found"

  face, scores = results
  face = cv2.resize(face, (256, 256))
  
  if args.save_face:
    cv2.imwrite("detected_face.png", face)

  for i, score in enumerate(scores):
    name, score = score
    print(f"{i+1}. {name}  score: {score:.5f}")


if __name__ == "__main__":
  parse = argparse.ArgumentParser("Demo of Face Recognition by KaiRA")
  parse.add_argument("image",       type=str,            help="path of input image")
  parse.add_argument("--save_face", action="store_true", help="save detected face image")

  args = parse.parse_args()
  main(args)
