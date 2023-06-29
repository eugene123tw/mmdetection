# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import numpy as np

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


def filter_blood_vessel(result, overlap_thres=1.0):
    # filter blood_vessel in glomerulus
    bbox_idx, mask_idx = 0, 1
    glomerulus_idx, blood_vessel_idx = 0, 1

    ious = bbox_overlaps(
        result[bbox_idx][blood_vessel_idx][:, :4],
        result[bbox_idx][glomerulus_idx][:, :4],
        mode='iof',
    )

    matched_blood_vessel_idx, matched_glomerulus_idx = np.where(
        ious == overlap_thres)
    if len(matched_blood_vessel_idx):
        keep = np.ones(len(result[bbox_idx][blood_vessel_idx]), dtype=bool)
        keep[matched_blood_vessel_idx] = False
        result[bbox_idx][blood_vessel_idx] = result[bbox_idx][
            blood_vessel_idx][keep]
        result[mask_idx][blood_vessel_idx] = [
            mask
            for keep, mask in zip(keep, result[mask_idx][blood_vessel_idx])
            if keep
        ]
    return result


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # test a single image
    result = inference_detector(model, args.img)

    result = filter_blood_vessel(result)

    show_result_pyplot(
        model,
        args.img,
        result,
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
