#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import collections
import numpy as np

from yolox.utils import assess_horizontal_or_vertical, intersect

counter = collections.OrderedDict()

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def plot_tracking(image, tlwhs, previous_tlwhs, obj_ids, line, scores=None,  fps_pro=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    text_scale = max(0.5, im_w/1600)
    text_thickness = int(text_scale*2)
    line_thickness = 2
    
    global counter 
    if 'up' not in counter:
        counter = {}
        counter['up'] = 0
        counter['down'] = 0
        counter['left'] = 0
        counter['right'] = 0
    else:
        pass
    horizontal_True_vertical_False = assess_horizontal_or_vertical(line)  
    
    # show FPS, avgFPS and number of detected people
    cv2.putText(im, 'FPS: %.2f - avgFPS: %.2f' % (fps_pro, fps),
                (int(im_w/40), int(im_h/10)), cv2.FONT_HERSHEY_COMPLEX, text_scale, (0, 0, 255), lineType=cv2.LINE_AA, thickness=text_thickness)

    # draw line counting
    cv2.line(im, line[0], line[1], (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    for i, tlwh in enumerate(tlwhs):
        # calculate current and previous center point of bboxes
        x1, y1, w, h = tlwh
        x2, y2, w2, h2 = previous_tlwhs[i]
        cur_center = (x1 + (w/2), y1 + (h/2))
        pre_center = (x2 + (w2/2), y2 + (h2/2))
        
        # counting person enter and exit
        if intersect(cur_center, pre_center, line[0], line[1]):
            if horizontal_True_vertical_False:
                if cur_center[1] < pre_center[1]:  
                    counter['up'] += 1  
                    cv2.line(im, line[0], line[1], (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                elif cur_center[1] > pre_center[1]: 
                    counter['down'] += 1
                    cv2.line(im, line[0], line[1], (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                else:
                    pass
            else:
                if cur_center[0] < pre_center[0]: 
                    counter['left'] += 1 
                    cv2.line(im, line[0], line[1], (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                elif cur_center[0] > pre_center[0]: 
                    counter['right'] += 1
                    cv2.line(im, line[0], line[1], (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                else:
                    pass

        # draw connection or trajectory of object during 2 consecutive frames
        cur = tuple(map(int, cur_center))
        pre = tuple(map(int, pre_center))
        cv2.line(im, cur, pre, color=(0, 0 ,255), thickness=line_thickness, lineType=cv2.LINE_AA)
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))

        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        (text_width, text_height), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_DUPLEX, 1, 1)  

        # draw bboxes
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.rectangle(im, (intbox[0], intbox[1]), (intbox[0]+text_width-1, intbox[1]+text_height-1), color, cv2.FILLED)
        cv2.putText(im, id_text, (intbox[0], intbox[1]+text_height-1), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), lineType=cv2.LINE_AA,
                    thickness=text_thickness)

    # show number 
    if horizontal_True_vertical_False:
        cv2.putText(im, 'Up: %d - Down: %d' % (counter['up'], counter['down']),
                (int(im_w/40), (int(im_h/10) + (int(text_scale*30)))), cv2.FONT_HERSHEY_COMPLEX, text_scale, (255, 0, 0), lineType=cv2.LINE_AA, thickness=text_thickness)
    else:
        cv2.putText(im, 'Left: %d - Right: %d' % (counter['left'], counter['right']),
                (int(im_w/40), (int(im_h/10) + (int(text_scale*30)))), cv2.FONT_HERSHEY_COMPLEX, text_scale, (255, 0, 0), lineType=cv2.LINE_AA, thickness=text_thickness)
    return im