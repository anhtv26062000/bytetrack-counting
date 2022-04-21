import os
import argparse

import cv2
import time
import copy
import onnx
import onnxruntime
import numpy as np

from utils.counts_per_sec import CountsPerSec

# hide onnxruntime warning 
onnxruntime.set_default_logger_severity(3)

from loguru import logger

from yolox.data.data_augment import preproc as preprocess
from yolox.utils import multiclass_nms, demo_postprocess, plot_tracking, hard_nms
from yolox.tracker.byte_tracker import BYTETracker

# for multi-processing
from multiprocessing import Process, Queue, Value

# Global variables
frame_id = 0
input_f_playback = Queue()
playing = Value('i', 1)
input_f_mot = Queue()
output_bboxs = Queue()
faces_output_bboxs = Queue()
# bk_bboxes = []
output_fps = Queue()
bk_fps = None
tracker = None

predictor = None
args = None

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
        # default=-1,
        default='samples/test4_unicam.mp4',
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
    parser.add_argument("--skip_frames", type=int, default=1, help="number frames for skipping")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.85, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

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
    global faces_output_bboxs
    bk_bboxes = None
    bk_face_bboxes = []
    no_face_bb_count = 0
    no_bb_count = 0
    global playing
    cps = CountsPerSec().start()
    counts_per_sec = 0.0
    print('@@ Start Playback input_f_playback: ', input_f_playback.qsize())
    try:
        # while playing.value or not input_f_playback.empty():
        while True:
            if(input_f_playback.empty()):
                continue
            frame = input_f_playback.get()
            # print('@@ output_bboxs.qsize(): ', output_bboxs.qsize())
            if (output_bboxs.qsize() > 0):
                counts_per_sec = cps.countsPerSec()
                print('Counts Per Sec: {:.2f} '.format(counts_per_sec))
                cps.increment()
                output_bbox = output_bboxs.get()
                if (output_bbox != None):
                    online_tlwhs, previous_tlwhs, online_ids, line, fps_pro, fps = output_bbox
                    bk_bboxes = (online_tlwhs, previous_tlwhs, online_ids, line, fps_pro, fps)
                    # print('@@ output_bboxs: ', online_tlwhs, previous_tlwhs, online_ids, line, fps_pro, fps)
                    frame = plot_tracking(
                        frame, online_tlwhs, previous_tlwhs, online_ids, line, counts_per_sec, fps, no_sum= False)  
            else:
                no_bb_count += 1
                if ( no_bb_count < 120 and bk_bboxes is not None):
                    online_tlwhs, previous_tlwhs, online_ids, line, fps_pro, fps = bk_bboxes
                    # print('@@ output_bboxs: ', online_tlwhs, previous_tlwhs, online_ids, line, fps_pro, fps)
                    frame = plot_tracking(
                        frame, online_tlwhs, previous_tlwhs, online_ids, line, counts_per_sec, fps, no_sum = True)
                else:
                    bk_bboxes = None
                    no_bb_count = 0 
            
            # print('@@ get_bb_fps: ', get_bb_fps)
            # print('@@ faces_output_bboxs: ', faces_output_bboxs.qsize())
            # if (faces_output_bboxs.qsize() > 0 ):
            #     face_bb = faces_output_bboxs.get()
            #     for i in face_bb:
            #         box = i
            #         cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)
            #     bk_face_bboxes = face_bb
            # else:
            #     no_face_bb_count += 1
            #     if (no_face_bb_count < 300 and len(bk_face_bboxes) > 0):
            #         for i in bk_face_bboxes:
            #             box = i
            #             cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)                
        
            cv2.imshow(win_name, frame)
            cv2.resizeWindow(win_name, 960, 540)
            
            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        playing.value = 0
        cv2.destroyAllWindows()
    except Exception as e:
        raise e  

def run_video_capture(args):
    cap = cv2.VideoCapture(args.input)
    # get_fps = cap.get(cv2.CAP_PROP_FPS)
    # print('@@ video fps: ', get_fps)
    # set_fps = cap.set(cv2.CAP_PROP_FPS, int(30))
    # print('@@ set_fps ', set_fps)
    
    # frame_rate = 15    
    # prev = 0

    while True:
        # time_elapsed = time.time() - prev

        ret_val, frame = cap.read()
        
        if ret_val:
            # if time_elapsed > 1./frame_rate:
                # prev = time.time()
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

def mot(worker_name):
    t1 = time.time()
    print('@@ Start MOT ', worker_name)
    global args
    # if (args == None):
    #    args = make_parser().parse_args()
    global predictor
    if (predictor == None):
        predictor = Predictor(args)

    width = 800
    height = 450
    line = [(0, int(height/15*7)), (int(width-1), int(height/15*4))]

    global tracker
    if (tracker is None):
        tracker = BYTETracker(args, frame_rate=30)

    # =============ultra light weight face detection load model=========================
    # onnx_path = "./pretrained/version-RFB-320.onnx"

    # threshold = 0.7

    # predictor_face = onnx.load(onnx_path)
    # onnx.checker.check_model(predictor_face)
    # onnx.helper.printable_graph(predictor_face.graph)

    # ort_session = onnxruntime.InferenceSession(onnx_path)
    # input_name = ort_session.get_inputs()[0].name
    # ==================================================================================

    global frame_id
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

    print('@@ elapsed t1: ', time.time() - t1)

    while True:
        if (input_f_mot.qsize() == 0):
            continue
        print('@@ MOT {} input_f_mot {}'.format(worker_name, str(input_f_mot.qsize())))
        # print('@@ input_f_mot: ', input_f_mot.qsize())
        start_time = time.time()
        frame = input_f_mot.get()

        # preprocess face detection
        # img_face = copy.deepcopy(frame)
        # img_face = cv2.resize(img_face, (320, 240))
        # image_mean = np.array([127, 127, 127])
        # img_face = (img_face - image_mean) / 128
        # img_face = np.transpose(img_face, [2, 0, 1])
        # img_face = np.expand_dims(img_face, axis=0)
        # img_face = img_face.astype(np.float32)

        # Skip frames
        if frame_id % args.skip_frames == 0:
            t2 = time.time()
            outputs, img_info = predictor.inference(frame)
            print('@@ t2: ', time.time() - t2)
            # print('@@ inference bytetrack_tiny: ', img_info)
            # t3 = time.time()
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
                # print('@@ t3: ', time.time() - t3)
                # calculate number of deteting and tracking frame per second 
                fps_process = 1 / (time.time() - start_time)
                print('@@ fps_process: ', fps_process)
                
                # fps = args.skip_frames / (time.time() - start_time)
                # print('@@ skip_frames fps: ', fps)

                dif = list(set(online_ids).symmetric_difference(set(previous_ids)))
                if len(dif) > 0:
                    for x in reversed(dif):
                        # print(previous_ids.index(x))
                        del previous_tlwhs[previous_ids.index(x)]
                        del previous_ids[previous_ids.index(x)]
                global output_bboxs
                if (output_bboxs.qsize() < 9):
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
            
            # Ultra-lightweight Face Detection
            # confidences, boxes = ort_session.run(None, {input_name: img_face})
            # boxes, labels, probs = predict(width, height, confidences, boxes, threshold)
            # global faces_output_bboxs
            # if (faces_output_bboxs.qsize() < 1):
            #     faces_output_bboxs.put(boxes)
            # calculate number of frame per "int(skip_frames)" seconds
            fps = args.skip_frames / (time.time() - start_time)

        else:
            if (output_bboxs.qsize() < 9 ):
                output_bboxs.put((online_tlwhs, previous_tlwhs, online_ids, line, fps_process, fps))
                # print('@@ Put out bboxs: ', online_tlwhs, previous_tlwhs, online_ids, line, fps_process, fps)

            # online_im = plot_tracking(
            #         frame, online_tlwhs, previous_tlwhs, online_ids, line, fps_pro=fps_process, fps=fps)  
        print('@@ ================ Frame: {} =============='.format(frame_id))        
        frame_id +=1
    

if __name__ == '__main__':
    # Start Playback Processing
    p1 = Process(target=playback, args=(), daemon=True)
    p1.start()

    # Start MOT Processing
    args = make_parser().parse_args()
    # predictor = Predictor(args)
    for i in range(1):
        worker_name = 'worker ' + str(i)
        p2 = Process(target=mot, args=(worker_name,), daemon=False)
        p2.start()

    # run capture video
    run_video_capture(args)