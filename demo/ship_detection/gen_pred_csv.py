import argparse
import json
import pickle

import numpy as np
import pandas as pd


def get_image_list(anno_json):
    with open(anno_json) as f:
        data = json.load(f)
    images = data['images']
    img_list = []
    for image in images:
        img_list.append(image['file_name'])
    return img_list


def gen_pred_csv(pickle_file, ann_json, submission_csv, sample_submission_csv):
    sub_image_list = pd.read_csv(sample_submission_csv)['id'].values

    with open(pickle_file, 'rb') as f:
        det_results = np.asarray(
            pickle.load(f), dtype=object)  # [(bg + cls), images]
    det_results = np.transpose(det_results,
                               (1, 0))  # dim should be (class, image)
    img_list = get_image_list(ann_json)
    img_list = [img_name.replace('.jpg', '.png') for img_name in img_list]

    # det_results index 0 as there's only one class
    img_det = {img_list[i]: det_results[0, i] for i in range(len(img_list))}
    rows = []
    for img_id in sub_image_list:
        det = img_det[img_id]
        bbox_str = ''
        selected = det[:, -1] > 0.0
        det = det[selected]
        if len(det) == 0:
            bbox_str = '0.0 0 0 0 0'
        else:
            for bbox in det:
                bbox = list(bbox)
                bbox_str += f'{bbox[4]:0.1f} {int(bbox[0])} {int(bbox[1])} {int(bbox[2])} {int(bbox[3])}, '
            bbox_str = bbox_str[:-2]
        rows.append(bbox_str)
    df = pd.DataFrame({'id': sub_image_list, 'label': rows})
    df.to_csv(submission_csv, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate prediction csv from result pickle')
    parser.add_argument('pickle_file', help='result pickle file')
    parser.add_argument('ann_json', help='annotation json file')
    parser.add_argument(
        '--sample_sub',
        default=
        '/home/yuchunli/_DATASET/ship-detection/.extras/sample_submission.csv')
    parser.add_argument(
        '--output', default='submission.csv', help='output csv file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    gen_pred_csv(args.pickle_file, args.ann_json, args.output, args.sample_sub)
