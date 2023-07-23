# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import pickle

import cv2
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import streamlit as st

from mmdet.core import PolygonMasks

CONFIG = 'work_dirs/mask2former_swin-s-p4-w7-224_lsj_s5_cls1/mask2former_swin-s-p4-w7-224_lsj_s5_cls1.py'
PICKLE = 'work_dirs/mask2former_swin-s-p4-w7-224_lsj_s5_cls1/val.pkl'


@st.cache_data
def parse_json(config_path: str, phase: str = 'val'):
    cfg = mmcv.Config.fromfile(config_path)
    json_file_path = cfg.data[phase].ann_file

    with open(json_file_path) as f:
        coco_json = json.load(f)
    category = [c['name'] for c in coco_json['categories']]  # 80 classes

    annotations = coco_json['annotations']
    images = coco_json['images']

    category_dict = {c['id']: c['name'] for c in coco_json['categories']}

    # id to image mapping
    image_dict = {}
    img_list = []
    img_id_list = []

    for image in images:
        key = image['id']
        image_dict[key] = [image['file_name']]
        img_list.append(image['file_name'])
        img_id_list.append(key)

    total_annotations = {}

    for anno in annotations:
        image_name = image_dict[anno['image_id']][0]
        idx = anno['category_id']
        single_ann = []
        single_ann.append(category_dict[idx])
        x1, y1, w, h = anno['bbox']
        x2, y2 = x1 + w, y1 + h
        single_ann.append([x1, y1, x2, y2])
        if 'segmentation' in anno:
            if isinstance(anno['segmentation'], list):
                single_ann.append(anno['segmentation'])
            else:
                rle = maskUtils.frPyObjects(anno['segmentation'],
                                            *anno['segmentation']['size'])
                mask = maskUtils.decode(rle)
                single_ann.append(mask)

        if image_name not in total_annotations:
            total_annotations[image_name] = []

        total_annotations[image_name].append(single_ann)

    return category, img_list, total_annotations


@st.cache_data
def get_det_results(pickle_file: str, category: list, image_list: list):
    with open(pickle_file, 'rb') as f:
        det_results = pickle.load(f)
    dets_dict = {}
    for i, det in enumerate(det_results):
        dets_dict[image_list[i]] = []
        if isinstance(det, tuple):
            boxes, masks = det
        else:
            # TODO: figure out this part
            boxes, masks = det, None
        for j, cls_boxes in enumerate(boxes):
            label = category[j]
            for n in range(len(cls_boxes)):
                single_det = []
                single_det.append(label)
                single_det.append(cls_boxes[n])
                if masks is not None:
                    mask = masks[j][n]
                else:
                    mask = None
                single_det.append(mask)
                dets_dict[image_list[i]].append(single_det)
    return dets_dict


@st.cache_data
def load_image(image_name: str, path_to_folder: str, bgr2rgb: bool = True):
    """Load the image
    Args:
        image_name (str): name of the image
        path_to_folder (str): path to the folder with image
        bgr2rgb (bool): converts BGR image to RGB if True
    """
    path_to_image = os.path.join(path_to_folder, image_name)
    image = cv2.imread(path_to_image)
    if bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_iou(self, det, gt):
    iou = [[] for _ in range(len(det))]

    for idx, cls_objs in enumerate(det):
        category = self.data_info.aug_category.category[idx]
        BBGT = []
        for t in gt:
            if not t[0] == category:
                continue
            BBGT.append([t[1], t[2], t[1] + t[3], t[2] + t[4]])
        BBGT = np.asarray(BBGT)
        d = [0] * len(BBGT)  # for check 1 GT map to several det

        confidence = cls_objs[:, 4]
        BB = cls_objs[:, 0:4]  # bounding box

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]

        # for returning original order
        ind_table = {i: sorted_ind[i] for i in range(len(sorted_ind))}

        for _ in range(len(BB)):
            iou[idx].append([])

        if len(BBGT) > 0:
            for i in range(len(BB)):
                bb = BB[i, :]

                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)  # max overlaps with all gt
                jmax = np.argmax(overlaps)

                if ovmax > self.iou_threshold:
                    if not d[jmax]:
                        d[jmax] = 1
                    else:  # multiple bounding boxes map to one gt
                        ovmax = -ovmax

                iou[idx][ind_table[i]] = ovmax  # return to unsorted order
    return iou


def draw_gt(image,
            selected_category,
            category,
            annotations,
            show_txt=True,
            gt_box_color=(255, 140, 0)):
    img_h, img_w, _ = image.shape

    color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)

    for anno in annotations:
        label = anno[0]
        if label != selected_category and selected_category != 'all':
            continue
        bbox = anno[1]
        mask = anno[2] if len(anno) == 3 else None

        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w - 1, x2)
        y2 = min(img_h - 1, y2)
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if show_txt:
            if y2 + 30 >= img_h:
                cv2.rectangle(image, (x1, y1),
                              (x1 + len(label) * 10, int(y1 - 20)),
                              gt_box_color, cv2.FILLED)
                cv2.putText(image, label, (x1, int(y1 - 5)), font, 0.5,
                            (255, 255, 255), 1)
            else:
                cv2.rectangle(image, (x1, y2),
                              (x1 + len(label) * 10, int(y2 + 20)),
                              gt_box_color, cv2.FILLED)
                cv2.putText(image, label, (x1, int(y2 + 15)), font, 0.5,
                            (255, 255, 255), 1)
        cv2.rectangle(image, (x1, y1), (x2, y2), gt_box_color, 1)
        if mask:
            if isinstance(mask[0], list):
                contours = np.array(mask).reshape(-1)
                contours = contours.astype(np.int32)
                polygon_masks = PolygonMasks([[contours]], img_h, img_w)
                bitmask = polygon_masks.to_ndarray()[0]
                image[bitmask] = image[bitmask] * 0.5 + color_mask * 0.5


def draw_pred(image,
              selected_category,
              category,
              predictions,
              score_thres=0.5,
              show_txt=True,
              det_box_color=(255, 255, 0)):
    color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
    img_h, img_w, _ = image.shape
    for pred in predictions:
        label = pred[0]
        x1, y1, x2, y2, score = pred[1]
        if label != selected_category and selected_category != 'all':
            continue
        if score < score_thres:
            continue
        mask = pred[2] if len(pred) == 3 else None
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w - 1, x2)
        y2 = min(img_h - 1, y2)
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        if show_txt:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = label + ' : ' + '{:.2f}'.format(score)

            if y2 + 30 >= img_h:
                cv2.rectangle(image, (x1, y1),
                              (x1 + len(text) * 9, int(y1 - 20)), (0, 0, 255),
                              cv2.FILLED)
                cv2.putText(image, text, (x1, int(y1 - 5)), font, 0.5,
                            (255, 255, 255), 1)
            else:
                cv2.rectangle(image, (x1, y2),
                              (x1 + len(text) * 9, int(y2 + 20)), (0, 0, 255),
                              cv2.FILLED)
                cv2.putText(image, text, (x1, int(y2 + 15)), font, 0.5,
                            (255, 255, 255), 1)

        cv2.rectangle(image, (x1, y1), (x2, y2), det_box_color, 2)
        if mask:
            if isinstance(mask, list):
                contours = np.array(mask).reshape(-1)
                contours = contours.astype(np.int32)
                polygon_masks = PolygonMasks([[contours]], img_h, img_w)
                bitmask = polygon_masks.to_ndarray()[0]
            elif isinstance(mask, np.ndarray):
                bitmask = mask
            elif isinstance(mask, dict):
                bitmask = maskUtils.decode(mask).astype(bool)

            image[bitmask] = image[bitmask] * 0.5 + color_mask * 0.5


config = mmcv.Config.fromfile(CONFIG)
category, img_list, total_annotations = parse_json(CONFIG)
dets = get_det_results(PICKLE, category, img_list)
st.title('Hubmap Demo')
st.sidebar.title('Hubmap Demo')
st.sidebar.markdown('This is a demo for Hubmap.')
selected_image = st.sidebar.selectbox('Select image', img_list)

image = load_image(selected_image, config.data.val.img_prefix)
draw_ground_truth = st.sidebar.checkbox('Draw ground truth')
draw_prediction = st.sidebar.checkbox('Draw prediction')
selected_category = st.sidebar.selectbox('Select category', category + ['all'])

score_thres = st.sidebar.slider('Score threshold', 0.0, 1.0, 0.5, 0.1)

if draw_ground_truth:
    draw_gt(image, selected_category, category,
            total_annotations[selected_image])
if draw_prediction:
    draw_pred(
        image,
        selected_category,
        category,
        dets[selected_image],
        score_thres=score_thres)

st.image(image, caption=selected_image, use_column_width=True)
