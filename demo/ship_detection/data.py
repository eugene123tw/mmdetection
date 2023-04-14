import math
import os

import cv2
import numpy as np
import pandas as pd
import PIL
from datumaro import AnnotationType, Bbox, DatasetItem, LabelCategories
from datumaro.components.project import Dataset
from datumaro.util.image import Image
from sklearn.model_selection import train_test_split, KFold

PIL.Image.MAX_IMAGE_PIXELS = 933120000


class ShipDetectionDataset:
    """Ship Detection from HuggingFace Dataset."""

    def __init__(self, data_root) -> None:
        self.data_root = data_root
        self.categories = {
            AnnotationType.label: LabelCategories.from_iterable(['ship'])
        }

    def make_coco_test(self, export_path):
        test_root = os.path.join(self.data_root, 'test')
        df = pd.read_csv(
            os.path.join(self.data_root, '.extras/sample_submission.csv'))
        anno_dict = {}

        for index, row in df.iterrows():
            image_id = row['id']
            if image_id not in anno_dict:
                anno_dict[image_id] = []
            anno_dict[image_id].append([0, 0, 100, 100])

        dsitems = []
        for index, (image_id, bboxes) in enumerate(anno_dict.items()):
            img = cv2.imread(os.path.join(test_root, f'{image_id}'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            attributes = {'filename': f'{self.data_root}/test/{image_id}.png'}
            image = Image(data=img, size=img.shape[0:2])
            datumaro_bboxes = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                box = Bbox(x=x1, y=y1, w=(x2 - x1), h=(y2 - y1), label=0)
                datumaro_bboxes.append(box)

            dsitems.append(
                DatasetItem(
                    id=index,
                    annotations=datumaro_bboxes,
                    subset='test',
                    image=image,
                    attributes=attributes))

        dataset = Dataset.from_iterable(dsitems, categories=self.categories)
        dataset.export(export_path, 'coco', default_image_ext='.png')

    def make_coco_train(self, export_path, fold=5):
        train_root = os.path.join(self.data_root, 'train')
        df = pd.read_csv(os.path.join(self.data_root, '.extras/train.csv'))
        anno_dict = {}

        for index, row in df.iterrows():
            image_id = row['id']
            if image_id not in anno_dict:
                anno_dict[image_id] = []
            anno_dict[image_id].append(
                [row['xmin'], row['ymin'], row['xmax'], row['ymax']])

        dsitems = []
        for index, (image_id, bboxes) in enumerate(anno_dict.items()):
            img = cv2.imread(os.path.join(train_root, f'{image_id}'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            attributes = {'filename': f'{self.data_root}/train/{image_id}.png'}
            image = Image(data=img, size=img.shape[0:2])
            datumaro_bboxes = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                box = Bbox(x=x1, y=y1, w=(x2 - x1), h=(y2 - y1), label=0)
                datumaro_bboxes.append(box)

            dsitems.append(
                DatasetItem(
                    id=index,
                    annotations=datumaro_bboxes,
                    image=image,
                    attributes=attributes))

        kf = KFold(n_splits=fold, shuffle=True, random_state=42)
        for fold, (train_indices, val_indices) in enumerate(kf.split(dsitems)):
            for index in train_indices:
                dsitems[index].subset = f'train'
            for index in val_indices:
                dsitems[index].subset = f'val'

            dataset = Dataset.from_iterable(dsitems, categories=self.categories)
            dataset.export(f"{export_path}-fold-{fold}", 'coco', default_image_ext='.png')

    def get_adaptive_tile_params(self, object_tile_ratio=0.01, rule='avg'):
        tile_cfg = dict(
            tile_size=None, tile_overlap=None, tile_max_number=None)
        bboxes = np.zeros((0, 4), dtype=np.float32)
        image_sizes = []
        max_object = 0
        for dataset_item in self.dataset['train']:
            image_sizes.extend([dataset_item['image'].size] *
                               len(dataset_item['objects']['bbox']))
            anno = dataset_item['objects']
            bbox = np.array(anno['bbox'])
            bboxes = np.concatenate((bboxes, bbox), 0)
            if len(bbox) > max_object:
                max_object = len(bbox)

        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

        if rule == 'min':
            object_area = np.min(areas)
        elif rule == 'avg':
            object_area = np.median(areas)

        # NOTE: another strategy is to compute the minimum detectable object size
        # For instance, in order to have > 1 pixel in the smallest feature map, the minimum object size should be VFNet the largest stride is 128

        max_area = np.max(areas)

        tile_size = int(math.sqrt(object_area / object_tile_ratio))
        overlap_ratio = max_area / (
            tile_size**2) if max_area / (tile_size**2) < 1.0 else None

        tile_cfg.update(dict(tile_size=tile_size, tile_max_number=max_object))
        if overlap_ratio:
            tile_cfg.update(dict(tile_overlap=overlap_ratio))
        print(tile_cfg)
        return tile_cfg


if __name__ == '__main__':
    dataset = ShipDetectionDataset(
        data_root='/home/yuchunli/_DATASET/ship-detection')
    # dataset.get_adaptive_tile_params(rule='min')
    # dataset.make_coco(export_path="/home/yuchunli/_DATASET/ship-detection-coco")
    # dataset.make_coco(export_path="/home/yuchunli/_DATASET/ship-detection-coco-full")
    # dataset.make_fold(export_path="/home/yuchunli/_DATASET/ship-detection-coco/ship-detection-coco")

    dataset.make_coco_train(
        export_path="/home/yuchunli/_DATASET/ship-detection-coco")

    # dataset.make_coco_test(
    #     root='/home/yuchunli/_DATASET/ship-detection',
    #     export_path='/home/yuchunli/_DATASET/ship-detection-coco-full-test')
