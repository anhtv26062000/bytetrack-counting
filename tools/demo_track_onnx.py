import argparse
import os
import sys
sys.path.insert(0, './yolov5face')

import cv2
import time
import copy
import torch
import onnxruntime
import numpy as np
from loguru import logger

from yolox.data.data_augment import preproc as preprocess
from yolox.utils import mkdir, multiclass_nms, demo_postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

from yolov5face.models.experimental import attempt_load
from yolov5face.utils.datasets import letterbox
from yolov5face.utils.general import check_img_size, non_max_suppression_face, scale_coords, scale_coords_landmarks, xyxy2xywh

def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="./pretrained/bytetrack_tiny.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default='./videos/palace.mp4',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='./YOLOX_outputs/onnx/',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.01,
        help="Score threshold to filter the result.",
    )
    parser.add_argument(
        "-n",
        "--nms_thr",
        type=float,
        default=0.6,
        help="NMS threshould.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="608,1088",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    # tracking args
    parser.add_argument("--skip_frames", type=int, default=15, help="number frames for skipping")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.83, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser

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

class Predictor(object):
    def __init__(self, args):
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225) 
        self.args = args
        self.session = onnxruntime.InferenceSession(args.model)
        self.input_shape = tuple(map(int, args.input_shape.split(',')))

    def inference(self, ori_img):
        img_info = {"id": 0}
        height, width = ori_img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = ori_img

        img, ratio = preprocess(ori_img, self.input_shape, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}
        output = self.session.run(None, ort_inputs)
        predictions = demo_postprocess(output[0], self.input_shape, p6=self.args.with_p6)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.args.nms_thr, score_thr=self.args.score_thr)
        if dets is None:
            return dets, img_info
        return dets[:, :-1], img_info


def imageflow_demo(predictor, args):
    cap = cv2.VideoCapture(args.input)
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    width = 800
    height = 450
    line = [(0, int(height/15*7)), (int(width-1), int(height/15*4))]

    fps = cap.get(cv2.CAP_PROP_FPS)

    save_folder = args.output_dir
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, args.input.split("/")[-1])
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    tracker = BYTETracker(args, frame_rate=30)

    weights = "./yolov5face/models/yolov5s-face.pt"
    # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    img_size = 320
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())

    frame_id = 0
    fps = 0
    fps_process = 0
    results = []
    online_tlwhs = []
    previous_tlwhs = []
    online_ids = []
    previous_ids = []
    online_scores = []
    online_targets = []
    previous_targets = []
    
    while True:
        # print("\nFrame ID: ", frame_id)
        # if frame_id % 20 == 0:
        #     logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        start_time = time.time()
        ret_val, frame = cap.read()

        if ret_val:
            # resize frame to enhance processing speed (interpolation NEAREST is the fastest in cv2 interpolation)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

            # YOLOv5 copy original image
            img0 = copy.deepcopy(frame)

            # Skip frames
            if frame_id % args.skip_frames == 0:
                outputs, img_info = predictor.inference(frame)
                if outputs is not None:
                    online_targets, previous_targets = tracker.update(outputs, [img_info['height'], img_info['width']], [img_info['height'], img_info['width']])
                    # print('{0:<50} {1:<50}'.format("Current: "+str(online_targets), "Previous: "+str(previous_targets)))
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    for t in (online_targets):
                        tlwh = t.tlwh
                        tid = t.track_id
                        vertical = tlwh[2] / tlwh[3] > 1.6
                        if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                        else:
                            print("**************ALERT FOR TO ANH NOW*****************")
                    # results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
                    # calculate number of deteting and tracking frame per second 
                    fps_process = 1 / (time.time() - start_time)

                    dif = list(set(online_ids).symmetric_difference(set(previous_ids)))
                    if len(dif) > 0:
                        for x in reversed(dif):
                            # print(previous_ids.index(x))
                            del previous_tlwhs[previous_ids.index(x)]
                            del previous_ids[previous_ids.index(x)]
                    online_im = plot_tracking(
                        img_info['raw_img'], online_tlwhs, previous_tlwhs, online_ids, line, fps_pro=fps_process, fps=fps)

                    previous_tlwhs = []
                    previous_ids = []
                    for pre_t in (previous_targets):
                        pre_tlwh = pre_t.tlwh
                        pre_tid = pre_t.track_id
                        vertical = pre_tlwh[2] / pre_tlwh[3] > 1.6
                        if pre_tlwh[2] * pre_tlwh[3] > args.min_box_area and not vertical:
                            previous_tlwhs.append(pre_tlwh)
                            previous_ids.append(pre_tid)
                        else:
                            print("**************ALERT FOR TO ANH NOW*****************")
                else:
                    online_im = img_info['raw_img']

                # YOLOv5Face
                r = img_size / max(width, height)  # resize image to img_size
                if r != 1:  # always resize down, only resize up if training with augmentation
                    interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
                img0 = cv2.resize(img0, (int(width * r), int(height * r)), interpolation=interp)
                # Face Detector
                imgsz = check_img_size(img_size, s=stride)  # check img_size
                img = letterbox(img0, new_shape=imgsz)[0]
                img = torch.from_numpy(img).to(device).permute(2, 0, 1)
                img = img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = model(img)[0]
                # Apply NMS
                pred = non_max_suppression_face(pred, 0.75, 0.5)
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
                            online_im = show_results(online_im, xywh, conf, landmarks)

                # calculate number of frame per "int(skip_frames)" seconds
                fps = args.skip_frames / (time.time() - start_time)

            else:
                online_im = plot_tracking(
                        frame, online_tlwhs, previous_tlwhs, online_ids, line, fps_pro=fps_process, fps=fps)   
            # show stream window
            cv2.imshow("Video", online_im)
            # save video stream with format mp4
            vid_writer.write(online_im)
            
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

        else:
            break
        frame_id += 1
    return frame_id
        

if __name__ == '__main__':
    args = make_parser().parse_args()
    predictor = Predictor(args)
    stime = time.time()
    num_frames = imageflow_demo(predictor, args)
    fps = num_frames/(time.time()-stime)
    logger.info(f"Average FPS in this video: {fps}")