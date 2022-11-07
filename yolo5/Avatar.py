from eyes import eyes_best
from eyebrows import eyebrows_best
from faceshape import faceshape_best
import urllib.request
import time
from PIL import Image


def User_Avatar(User_img_path):
  result = faceshape_best(User_img_path)
  result = eyebrows_best(User_img_path, result)
  result = eyes_best(User_img_path, result)
  return result

# User_img = 'yolo5/000043.jpg'
# print(User_Avatar(User_img))

def img_download(url):
    img_path = "./yolo5/img/test.jpg"
    # 이미지 요청 및 다운로드
    urllib.request.urlretrieve(url, img_path)
    return img_path
