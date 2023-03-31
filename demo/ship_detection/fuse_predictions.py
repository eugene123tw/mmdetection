import argparse
import json
import pickle

import cv2
import numpy as np
from ensemble_boxes_wbf import weighted_boxes_fusion


def read_bboxes(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def fuse_prediction(pkl_paths, output, json_anno):
    with open(json_anno, 'rb') as f:
        anno = json.loads(f.read())

    img_dims = []
    for img in anno['images']:
        fname = img['file_name']
        img = cv2.imread(
            f'/home/yuchunli/_DATASET/ship-detection/test/{fname}')
        h, w, c = img.shape
        img_dims.append([w, h, w, h])
    model_level = []
    for i, pkl_path in enumerate(pkl_paths):
        boxes = read_bboxes(pkl_path)
        model_level.append(boxes)

    num_imgs = len(model_level[0])
    num_models = len(model_level)
    fused_boxes = []
    fused_scores = []
    for i in range(num_imgs):
        boxes = []
        scores = []
        labels = []
        for j in range(num_models):
            # NOTE: normalize bbox to [0, 1]
            boxes.append(model_level[j][i][0][:, :4] / np.array(img_dims[i]))
            scores.append(model_level[j][i][0][:, 4])
            labels.append(len(model_level[j][i][0]) * [0])
        fused_box, fused_score, fused_label = weighted_boxes_fusion(
            boxes,
            scores,
            labels,
            weights=None,
            iou_thr=0.5,
            skip_box_thr=0.0001)
        fused = np.concatenate(
            [fused_box * np.array(img_dims[i]),
             fused_score.reshape(-1, 1)],
            axis=1)
        fused_boxes.append(fused)

    output_boxes = []
    for fused_box in fused_boxes:
        output_boxes.append([fused_box])

    with open(args.output, 'wb') as f:
        pickle.dump(output_boxes, f)
    print(f'Output fused boxes to: {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction', nargs='+', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument(
        '--json',
        type=str,
        default=
        '/home/yuchunli/_DATASET/ship-detection-coco-full/annotations/instances_test.json'
    )
    args = parser.parse_args()
    fuse_prediction(args.prediction, args.output, args.json)
