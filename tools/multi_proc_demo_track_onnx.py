import argparse
import os
import sys
sys.path.insert(0, './yolov5face')

# for multi-processing
from multiprocessing import Process, Queue, Value

from PIL import Image

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

input_f_playback = Queue()
playing = Value('i', 1)
input_f_mot = Queue()
output_bboxs = Queue()
bk_bboxes = []
output_fps = Queue()
bk_fps = None

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
        default=-1,
        # default='samples/u_frontcam_cuted_853x480.mp4',
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

def playback():
    win_name = "output"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
    global input_f_playback
    global output_bboxs
    bk_bboxes = None
    no_bb_count = 0
    global playing
    print('@@ Start Playback input_f_playback: ', input_f_playback.qsize())
    try:
        while playing.value or not input_f_playback.empty():
            frame = input_f_playback.get()
            # print('@@ output_bboxs.qsize(): ', output_bboxs.qsize())
            if (output_bboxs.qsize() > 0):
                output_bbox = output_bboxs.get()
                if (output_bbox != None):
                    online_tlwhs, previous_tlwhs, online_ids, line, fps_pro, fps = output_bbox
                    bk_bboxes = (online_tlwhs, previous_tlwhs, online_ids, line, fps_pro, fps)
                    # print('@@ output_bboxs: ', online_tlwhs, previous_tlwhs, online_ids, line, fps_pro, fps)
                    frame = plot_tracking(
                        frame, online_tlwhs, previous_tlwhs, online_ids, line, fps_pro, fps)  
            else:
                no_bb_count += 1
                if ( no_bb_count < 300 and bk_bboxes is not None):
                    online_tlwhs, previous_tlwhs, online_ids, line, fps_pro, fps = bk_bboxes
                    # print('@@ output_bboxs: ', online_tlwhs, previous_tlwhs, online_ids, line, fps_pro, fps)
                    frame = plot_tracking(
                        frame, online_tlwhs, previous_tlwhs, online_ids, line, fps_pro, fps)
                else:
                    bk_bboxes = None
                    no_bb_count = 0  
        
            cv2.imshow(win_name, frame)
            cv2.resizeWindow(win_name, 960, 540)
            
            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        playing.value = 0
        cv2.destroyAllWindows()
    except Exception as e:
        raise e  

def mot(predictor, args):
    print('@@ Start MOT processing')
   
    width = 800
    height = 450
    line = [(0, int(height/15*7)), (int(width-1), int(height/15*4))]

    tracker = BYTETracker(args, frame_rate=30)

    weights = "./yolov5face/models/yolov5s-face.pt"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    global input_f_mot

    while not input_f_mot.empty():
        # print('@@ input_f_mot: ', input_f_mot.qsize())
        start_time = time.time()
        frame = input_f_mot.get()

        # YOLOv5 copy original image
        img0 = copy.deepcopy(frame)

        # Skip frames
        if frame_id % args.skip_frames == 0:
            outputs, img_info = predictor.inference(frame)
            # print('@@ inference bytetrack_tiny: ', img_info)
            if outputs is not None:
                online_targets, previous_targets = tracker.update(outputs, [img_info['height'], img_info['width']], [img_info['height'], img_info['width']])
                # print('@@ tracker update: {0:<50} {1:<50}'.format("Current: "+str(online_targets), "Previous: "+str(previous_targets)))
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
                # calculate number of deteting and tracking frame per second 
                fps_process = 1 / (time.time() - start_time)
                print('@@ fps_process: ', fps_process)
                fps = args.skip_frames / (time.time() - start_time)
                # print('@@ skip_frames fps: ', fps)

                dif = list(set(online_ids).symmetric_difference(set(previous_ids)))
                if len(dif) > 0:
                    for x in reversed(dif):
                        # print(previous_ids.index(x))
                        del previous_tlwhs[previous_ids.index(x)]
                        del previous_ids[previous_ids.index(x)]
                global output_bboxs
                if (output_bboxs.qsize() <= 1):
                    output_bboxs.put((online_tlwhs, previous_tlwhs, online_ids, line, fps_process, fps))
                    # print('@@ Put out bboxs: ', online_tlwhs, previous_tlwhs, online_ids, line, fps_process, fps)
                # online_im = plot_tracking(
                #     img_info['raw_img'], online_tlwhs, previous_tlwhs, online_ids, line, fps_pro=fps_process, fps=fps)

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
            # else:
            #     # if (output_bboxs.qsize() < 1):
            #     #     output_bboxs.put((online_tlwhs, previous_tlwhs, online_ids, line, fps_process, fps))
            #     # print('@@ Put out bboxs: ', online_tlwhs, previous_tlwhs, online_ids, line, fps_process, fps)
            #     output_bboxs.put(None)
            #     # online_im = img_info['raw_img']

            # calculate number of frame per "int(skip_frames)" seconds
            fps = args.skip_frames / (time.time() - start_time)

        else:
            if (output_bboxs.qsize() <= 1):
                output_bboxs.put((online_tlwhs, previous_tlwhs, online_ids, line, fps_process, fps))
                # print('@@ Put out bboxs: ', online_tlwhs, previous_tlwhs, online_ids, line, fps_process, fps)

            # online_im = plot_tracking(
            #         frame, online_tlwhs, previous_tlwhs, online_ids, line, fps_pro=fps_process, fps=fps)   
    

def run_video_capture(args):
    cap = cv2.VideoCapture(args.input)
    get_fps = cap.get(cv2.CAP_PROP_FPS)
    print('@@ video fps: ', get_fps)
    # set_fps = cap.set(cv2.CAP_PROP_FPS, int(30))
    # print('@@ set_fps ', set_fps)
   
    while True:
        ret_val, frame = cap.read()
        
        if ret_val:
            width = 800
            height = 450
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

            # Put frame to MOT processing
            global input_f_mot
            if (input_f_mot.qsize() < 1):
                input_f_mot.put(frame)
            
            # Put frame to Playback processing
            global input_f_playback
            if (input_f_playback.qsize() < 1):
                input_f_playback.put(frame)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    # Start Playback Processing
    p1 = Process(target=playback, args=())
    p1.start()

    # Start MOT Processing
    args = make_parser().parse_args()
    predictor = Predictor(args)
    p2 = Process(target=mot, args=(predictor, args))
    p2.start()

    # run capture video
    run_video_capture(args)

    # time.sleep(180)
    # if (playing.value == 0):
        # cv2.destroyAllWindows()
        # p1.join()
        # p2.join()