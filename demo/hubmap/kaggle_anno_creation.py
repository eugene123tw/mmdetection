import itertools
import json
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

df = pd.read_csv(
    '/home/yuchunli/_DATASET/hubmap-hacking-the-human-vasculature/tile_meta.csv'
)
df.head()

train_ids = df.query('dataset == 2')['id'].values.tolist()
valid_ids = df.query('dataset == 1')['id'].values.tolist()
print(len(train_ids), len(valid_ids))

jsonl_file_path = '/home/yuchunli/_DATASET/hubmap-hacking-the-human-vasculature/polygons.jsonl'
data = []
with open(jsonl_file_path, 'r') as file:
    for line in file:
        data.append(json.loads(line))


def coordinates_to_masks(coordinates, shape):
    masks = []
    for coord in coordinates:
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(coord)], 1)
        masks.append(mask)
    return masks


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(
            itertools.groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


coco_data = {
    'info': {},
    'licenses': [],
    'categories': [],
    'images': [],
    'annotations': []
}

categories = []
for item in tqdm(data, dynamic_ncols=True):
    annotations = item['annotations']
    for annotation in annotations:
        annotation_type = annotation['type']
        if annotation_type not in categories:
            categories.append(annotation_type)
            coco_data['categories'].append({
                'id': len(categories),
                'name': annotation_type
            })

train_coco_data = deepcopy(coco_data)
valid_coco_data = deepcopy(coco_data)

for item in tqdm(data, dynamic_ncols=True):
    image_id = item['id']

    if image_id in train_ids:
        ds = train_coco_data
    elif image_id in valid_ids:
        ds = valid_coco_data
    else:
        raise NotImplementedError()

    image_info = {
        'id': image_id,
        'file_name': item['id'] + '.tif',
        'height': 512,
        'width': 512
    }
    ds['images'].append(image_info)

    for annotation in item['annotations']:
        category_id = categories.index(annotation['type']) + 1

        segmentation = annotation['coordinates']
        mask_img = coordinates_to_masks(segmentation, (512, 512))[0]

        ys, xs = np.where(mask_img)
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        rle = binary_mask_to_rle(mask_img)

        annotation_info = {
            'id': len(ds['annotations']) + 1,
            'image_id': image_id,
            'category_id': category_id,
            'segmentation': rle,
            'bbox': [int(x1),
                     int(y1),
                     int(x2 - x1 + 1),
                     int(y2 - y1 + 1)],
            'area': int(np.sum(mask_img)),
            'iscrowd': 0,
        }
        ds['annotations'].append(annotation_info)

output_file_path = 'coco_annotations_train_all.json'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(train_coco_data, output_file, ensure_ascii=True, indent=4)

output_file_path = 'coco_annotations_valid_all.json'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(valid_coco_data, output_file, ensure_ascii=True, indent=4)
