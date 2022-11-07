from yolo5.eyes import eyes_best
from yolo5.eyebrows import eyebrows_best
from yolo5.faceshape import faceshape_best


def User_Avatar(User_img):
  result = faceshape_best(User_img)
  result = eyebrows_best(User_img, result)
  result = eyes_best(User_img, result)
  return result

User_img = 'yolo5/000043.jpg'
print(User_Avatar(User_img))