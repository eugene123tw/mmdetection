import json
import os

import cv2
import numpy as np
import pandas as pd
from datumaro import DatasetItem, Polygon
from datumaro.components.project import Dataset
from sklearn.model_selection import KFold

# Tiles from Dataset 1 have annotations that have been expert reviewed.
# Tiles from Dataset 2 contains sparse annotations that have NOT been expert reviewed.

# All of the test set tiles are from Dataset 1 (reviewed by experts).
# The training annotations contains Dataset 2 tiles from the public test WSI, but not from the private test WSI.
# Two of the WSIs make up the training set, two WSIs make up the public test set, and one WSI makes up the private test set.


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
                    id=index,
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


if __name__ == '__main__':
    dataset = HuBMAPVasculatureDataset(
        data_root='/home/yuchunli/_DATASET/hubmap-hacking-the-human-vasculature'
    )
    # dsitems = dataset.strategy_1()
    # dsitems = dataset.strategy_2()
    dsitems = dataset.strategy_4()
    dataset.export(
        dsitems,
        export_path='/home/yuchunli/_DATASET/HuBMAP-vasculature-coco-strategy_4'
    )
