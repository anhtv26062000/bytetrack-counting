import os
import argparse

import cv2
import time
import copy
import onnx
import onnxruntime
import numpy as np

# hide onnxruntime warning 
onnxruntime.set_default_logger_severity(3)

from loguru import logger

from yolox.data.data_augment import preproc as preprocess
from yolox.utils import multiclass_nms, demo_postprocess, plot_tracking, hard_nms
from yolox.tracker.byte_tracker import BYTETracker

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
    parser.add_argument("--skip_frames", type=int, default=10, help="number frames for skipping")
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

    # =============ultra light weight face detection load model=========================
    label_path = "./pretrained/voc-model-labels.txt"
    onnx_path = "./pretrained/onnx/version-RFB-320.onnx"
    class_names = [name.strip() for name in open(label_path).readlines()]

    threshold = 0.9

    predictor_face = onnx.load(onnx_path)
    onnx.checker.check_model(predictor_face)
    onnx.helper.printable_graph(predictor_face.graph)

    ort_session = onnxruntime.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    # ==================================================================================

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
        start_time = time.time()
        ret_val, frame = cap.read()

        if ret_val:
            # resize frame to enhance processing speed (interpolation NEAREST is the fastest in cv2 interpolation)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

            # copy image
            img_face = copy.deepcopy(frame)
            img_face = cv2.resize(img_face, (320, 240))

            # Skip frames
            if frame_id % args.skip_frames == 0:
                outputs, img_info = predictor.inference(frame)
                if outputs is not None:
                    online_targets, previous_targets = tracker.update(outputs, [img_info['height'], img_info['width']], [img_info['height'], img_info['width']])
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

                    # calculate number of processing frame per second 
                    fps_process = 1 / (time.time() - start_time)

                    dif = list(set(online_ids).symmetric_difference(set(previous_ids)))
                    if len(dif) > 0:
                        for x in reversed(dif):
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

                # Ultra-lightweight Detection
                image_mean = np.array([127, 127, 127])
                img_face = (img_face - image_mean) / 128
                img_face = np.transpose(img_face, [2, 0, 1])
                img_face = np.expand_dims(img_face, axis=0)
                img_face = img_face.astype(np.float32)
                confidences, boxes = ort_session.run(None, {input_name: img_face})
                boxes, labels, probs = predict(width, height, confidences, boxes, threshold)
                for i in range(boxes.shape[0]):
                    box = boxes[i, :]
                    cv2.rectangle(online_im, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)

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