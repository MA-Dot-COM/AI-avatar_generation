import argparse
import os
import platform
import sys
from pathlib import Path

import torch


from yolo5.models.common import DetectMultiBackend
from yolo5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolo5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolo5.utils.plots import Annotator, colors, save_one_box
from yolo5.utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()
def eyes_best(User_img, result):
  weights= ['yolo5/eyes_best.pt']  # model path or triton URL
  source = User_img  # file/dir/URL/glob/screen/0(webcam)
  data= 'coco128.yaml'  # dataset.yaml path
  imgsz=(640, 640) # inference size (height, width)
  conf_thres=0.1  # confidence threshold
  iou_thres=0.45  # NMS IOU threshold
  max_det=1000  # maximum detections per image
  device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
  view_img=False  # show results
  save_txt=False  # save results to *.txt
  save_conf=False  # save confidences in --save-txt labels
  save_crop=False  # save cropped prediction boxes
  nosave=False  # do not save images/videos
  classes=None  # filter by class: --class 0, or --class 0 2 3
  agnostic_nms=False  # class-agnostic NMS
  augment=False  # augmented inference
  visualize=False  # visualize features
  update=False  # update all models
  project='/content/'  # save results to project/name
  # name='exp'  # save results to project/name
  exist_ok=False  # existing project/name ok, do not increment
  line_thickness=3  # bounding box thickness (pixels)
  hide_labels=False  # hide labels
  hide_conf=False  # hide confidences
  half=False  # use FP16 half-precision inference
  dnn=False,  # use OpenCV DNN for ONNX inference
  vid_stride=1  # video frame-rate stride

  source = str(source)
  save_img = not nosave and not source.endswith('.txt')  # save inference images
  is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
  is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
  webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
  screenshot = source.lower().startswith('screen')
  if is_url and is_file:
      source = check_file(source)  # download

  # Load model
  device = select_device(device)
  model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
  stride, names, pt = model.stride, model.names, model.pt
  imgsz = check_img_size(imgsz, s=stride)  # check image size

  # Dataloader
  bs = 1  # batch_size
  dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
  vid_path, vid_writer = [None] * bs, [None] * bs

  # Run inference
  model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
  seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
  for path, im, im0s, vid_cap, s in dataset:
      with dt[0]:
          im = torch.from_numpy(im).to(model.device)
          im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
          im /= 255  # 0 - 255 to 0.0 - 1.0
          if len(im.shape) == 3:
              im = im[None]  # expand for batch dim

      # Inference
      with dt[1]:
          pred = model(im, augment=augment, visualize=visualize)

      # NMS
      with dt[2]:
          pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

      # Process predictions
      for i, det in enumerate(pred):  # per image
          seen += 1
          p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
          p = Path(p)  # to Path
          s += '%gx%g ' % im.shape[2:]  # print string
          if len(det):
              # Rescale boxes from img_size to im0 size
              det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

              # Print results

              max_conf = 0
              max_class = 0
              for conf_number, c in zip(det[:, 4], det[:, 5].unique()):
                  if max_conf < conf_number:
                      n = (det[:, 5] == c).sum()  # detections per class
                      s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                      max_class = c
                      max_conf = conf_number

              result.append(int(max_class))
  return result
