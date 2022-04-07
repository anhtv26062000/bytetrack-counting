# -*- coding: UTF-8 -*-
import argparse
import time

import cv2
import torch
import copy

import onnxruntime
import numpy as np

import sys
sys.path.insert(0, './yolov5face')

from yolov5face.models.experimental import attempt_load
from yolov5face.utils.datasets import letterbox
from yolov5face.utils.general import check_img_size, non_max_suppression_face, scale_coords, xyxy2xywh
from yolov5face.utils.torch_utils import time_synchronized

def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xywh, conf, landmarks):
    h,w,c = img.shape
    # tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    tl = 3
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def detect_stream(weights, source):
    # Load model
    img_size = 640
    conf_thres = 0.8
    iou_thres = 0.5

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu" 
    # stride = 
    providers =  ['CPUExecutionProvider']
    w = str(weights[0] if isinstance(weights, list) else weights)
    session = onnxruntime.InferenceSession(w, providers=providers)

    frame_count = 0
    detector_frame_skip = 1
    
    begin_frame_time = time.time()
    prev_frame_time = 0
    cap = cv2.VideoCapture(source)

    while(cap.isOpened()):
        
        ret, frame = cap.read() #BGR

        if frame_count % detector_frame_skip == 0:
            img0 = copy.deepcopy(frame)
            # h0, w0 = frame.shape[:2]  # orig hw
            # r = img_size / max(h0, w0)  # resize image to img_size
            # if r != 1:  # always resize down, only resize up if training with augmentation
            #     interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            # img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

            if ret == True:
                # imgsz = check_img_size(img_size, s=stride)  # check img_size
                # img = letterbox(img0, new_shape=imgsz)[0]
                # print(img.shape)
                img = torch.from_numpy(img0).to(device).permute(2, 0, 1)
                img = img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                print("img: ", img)
                print("imgshape: ", img.shape)

                # Inference
                im = img.cpu().numpy().astype(np.float32) # torch to numpy
                print(im.shape)
                pred = session.run(None, {session.get_inputs()[0].name: im})[0]
                print(pred)
                # Apply NMS
                pred = non_max_suppression_face(pred, conf_thres, iou_thres)

                # Process detections
                for det in pred:  # detections per image
                    gn = torch.tensor(frame.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
                    gn_lks = torch.tensor(frame.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                        det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], frame.shape).round()

                        for j in range(det.size()[0]):
                            xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                            conf = det[j, 4].cpu().numpy()
                            landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                            frame = show_results(frame, xywh, conf, landmarks)
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, 'FPS: %.2f' % (fps), (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
        cv2.imshow("To Anh dep trai:", frame)
        if cv2.waitKey(2) & 0xFF ==ord('q'):
            break
        frame_count += 1
    print("Avarage FPS:", frame_count / (time.time()- begin_frame_time))
    cap.release()
    cv2.destroyAllWindows()       



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp5/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--image', type=str, default=0, help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)

    detect_stream(opt.weights, opt.image)
