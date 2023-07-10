import glob
import json
import os

import cv2
import mmcv
import numpy as np
import pandas as pd
from datumaro import DatasetItem, Polygon
from datumaro.components.project import Dataset
from sklearn.model_selection import KFold

from demo.hubmap.single_model_test import initialize_detector
from mmdet.apis import inference_detector, init_detector

# Tiles from Dataset 1 have annotations that have been expert reviewed.
# Tiles from Dataset 2 contains sparse annotations that have NOT been expert reviewed.

# All of the test set tiles are from Dataset 1 (reviewed by experts).
# The training annotations contains Dataset 2 tiles from the public test WSI, but not from the private test WSI.
# Two of the WSIs make up the training set, two WSIs make up the public test set, and one WSI makes up the private test set.

BBOX_INDEX = 0
MASK_INDEX = 1
LABLE_INDEX = 1  # ONLY PICK BLOOD VESSEL


def binary_mask_to_polygon(binary_mask):
    contours, hierarchies = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchies is None:
        return []
    polygons = []
    areas = []
    for contour, hierarchy in zip(contours, hierarchies[0]):
        # skip inner contours
        if hierarchy[3] != -1 or len(contour) <= 2:
            continue
        areas.append(cv2.contourArea(contour))
        polygons.append(contour.flatten())

    if len(polygons) > 1:
        return polygons[np.argmax(areas)]
    return polygons[0]


class HuBMAPVasculatureDataset:

    def __init__(self, data_root) -> None:
        self.data_root = data_root
        # self.labels = ['glomerulus', 'blood_vessel', 'unsure']
        # self.labels = ['blood_vessel']
        self.labels = ['glomerulus', 'blood_vessel']
        self.df = pd.read_csv(os.path.join(self.data_root, 'tile_meta.csv'))
        np.random.seed(42)
        self.dsitem_dict = self._make_dsitems()

    def _make_dsitems(self):
        train_root = os.path.join(self.data_root, 'train')
        with open(os.path.join(self.data_root, 'polygons.jsonl'),
                  'r') as json_file:
            results = list(json_file)

        dsitem_dict = {}
        for index, result in enumerate(results):
            result = json.loads(result)
            image_id = result['id']
            img = cv2.imread(os.path.join(train_root, f'{image_id}.tif'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            attributes = {'filename': f'{self.data_root}/train/{image_id}.tif'}
            datumaro_polygons = []

            for anno in result['annotations']:
                if anno['type'] not in self.labels:
                    continue

                label_idx = self.labels.index(anno['type'])
                polygon = Polygon(
                    points=np.array(anno['coordinates']).flatten(),
                    label=label_idx,
                    z_order=0,
                    attributes=attributes)
                datumaro_polygons.append(polygon)

            if len(datumaro_polygons):
                dsitem_dict[image_id] = DatasetItem(
                    id=image_id,
                    annotations=datumaro_polygons,
                    image=img,
                    attributes=attributes)
        return dsitem_dict

    def strategy_1(self):
        """Train on Dataset 1, test on Dataset 2."""
        dsitems = []
        for index, row in self.df.iterrows():
            if self.dsitem_dict.get(row['id']) is not None:
                dsitem = self.dsitem_dict[row['id']]
                if row['dataset'] == 1:
                    dsitem.subset = 'train'
                elif row['dataset'] == 2:
                    dsitem.subset = 'val'
                dsitems.append(dsitem)
        return dsitems

    def strategy_2(self):
        """Train on Dataset 2, test on Dataset 1."""
        dsitems = []
        for index, row in self.df.iterrows():
            if self.dsitem_dict.get(row['id']) is not None:
                dsitem = self.dsitem_dict[row['id']]
                if row['dataset'] == 2:
                    dsitem.subset = 'train'
                elif row['dataset'] == 1:
                    dsitem.subset = 'val'
                dsitems.append(dsitem)
        return dsitems

    def strategy_3(self):
        """Train on Dataset 1, test on Dataset 1."""
        pass

    def strategy_4(self):
        """Train on WSI_1 (Dataset 1 + Dataset 2) , test on WSI_2 (Dataset
        1)"""
        dsitems = []
        for index, row in self.df.iterrows():
            if self.dsitem_dict.get(row['id']) is not None:
                dsitem = self.dsitem_dict[row['id']]
                if row['source_wsi'] == 1:
                    dsitem.subset = 'train'
                    dsitems.append(dsitem)
                elif row['source_wsi'] == 2 and row['dataset'] == 1:
                    dsitem.subset = 'val'
                    dsitems.append(dsitem)
        return dsitems

    def strategy_5(self):
        """Train on wsi 3, 4 + dataset 2, test on wsi 1,2 + dataset 1."""
        dsitems = []
        for index, row in self.df.iterrows():
            if self.dsitem_dict.get(row['id']) is not None:
                dsitem = self.dsitem_dict[row['id']]
                if row['dataset'] == 2:
                    if row['source_wsi'] == 3 or row['source_wsi'] == 4:
                        dsitem.subset = 'train'
                        dsitems.append(dsitem)

                if row['dataset'] == 1:
                    if row['source_wsi'] == 1 or row['source_wsi'] == 2:
                        dsitem.subset = 'val'
                        dsitems.append(dsitem)
        return dsitems

    def make_coco_split(self, dsitems, n_folds=5):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        folds = []
        for fold, (train_indices, val_indices) in enumerate(kf.split(dsitems)):
            for index in train_indices:
                dsitems[index].subset = 'train'
            for index in val_indices:
                dsitems[index].subset = 'val'
            folds.append(dsitems)
        return folds

    def export(self, dsitems, export_path):
        dataset = Dataset.from_iterable(dsitems, categories=self.labels)
        dataset.export(
            f'{export_path}',
            'coco',
            default_image_ext='.tif',
            save_media=True)

    def _get_unannotated_images(self):
        image_list = []
        for index, row in self.df.iterrows():
            if row['dataset'] == 3:
                image_list.append(row)
        return image_list

    def generate_pseudo_labeling(
        self,
        config,
        ckpt,
        iou_thr=0.5,
        nms_score_thr=0.001,
        max_num=100,
        score_thr=0.8,
    ):
        image_list = self._get_unannotated_images()

        config = mmcv.Config.fromfile(config)
        detector = initialize_detector(config, ckpt, iou_thr, nms_score_thr,
                                       max_num)
        dsitems = []
        for image in image_list:
            image_path = glob.glob(self.data_root + f'/train/{image["id"]}*')
            image_path = image_path[0]

            image_id = image['id']
            attributes = {'filename': image_path}
            datumaro_polygons = []
            result = inference_detector(detector, image_path)
            bboxes, masks = result
            bboxes, masks = bboxes[LABLE_INDEX], masks[LABLE_INDEX]
            num_predictions = bboxes.shape[0]
            for i in range(num_predictions):
                mask = masks[i]
                score = bboxes[i][-1]
                if score >= score_thr and mask.sum() > 0:
                    # NOTE: add dilation to make the mask larger
                    mask = mask.astype(np.uint8)
                    kernel = np.ones(shape=(3, 3), dtype=np.uint8)
                    bitmask = cv2.dilate(mask, kernel, 3)
                    bitmask = bitmask.astype(bool)
                    polygon = binary_mask_to_polygon(bitmask)
                    polygon = Polygon(
                        points=np.array(polygon).flatten(),
                        label=LABLE_INDEX,
                        z_order=0,
                        attributes=attributes)
                    datumaro_polygons.append(polygon)
            if len(datumaro_polygons):
                dsitems.append(
                    DatasetItem(
                        id=image_id,
                        annotations=datumaro_polygons,
                        image=image_path,
                        attributes=attributes))

        for dsitem in dsitems:
            dsitem.subset = 'train'
        return dsitems


if __name__ == '__main__':
    dataset = HuBMAPVasculatureDataset(
        data_root='/home/yuchunli/_DATASET/hubmap-hacking-the-human-vasculature'
    )
    # dsitems = dataset.strategy_1()
    # dsitems = dataset.strategy_2()
    # dsitems = dataset.strategy_5()
    # dataset.export(
    #     dsitems,
    #     export_path='/home/yuchunli/_DATASET/HuBMAP-vasculature-coco-strategy_5'
    # )

    dsitems = dataset.generate_pseudo_labeling(
        config=
        'work_dirs/solov2_x101_dcn_fpn_hubmap_s5_album_simple_aug/solov2_x101_dcn_fpn_hubmap_s5_aug.py',
        ckpt=
        'work_dirs/solov2_x101_dcn_fpn_hubmap_s5_album_simple_aug/best_segm_mAP_epoch_15.pth',
        iou_thr=0.6,
        nms_score_thr=0.65,
        max_num=100,
        score_thr=0.65,
    )
    dataset.export(
        dsitems,
        export_path=
        '/home/yuchunli/_DATASET/HuBMAP-vasculature-coco-pseudo-labeling')
