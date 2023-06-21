import os
import json
import numpy as np
import cv2
from datumaro import DatasetItem, Polygon
from datumaro.components.project import Dataset
from sklearn.model_selection import KFold
import pandas as pd


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

    def make_coco_train(self, export_path):
        train_root = os.path.join(self.data_root, 'train')
        with open(os.path.join(self.data_root, 'polygons.jsonl'), 'r') as json_file:
            results = list(json_file)

        dsitems = []
        for index, result in enumerate(results):
            result = json.loads(result)
            image_id = result['id']
            img = cv2.imread(os.path.join(train_root, f'{image_id}.tif'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            attributes = {'filename': f'{self.data_root}/train/{image_id}.tif'}
            datumaro_polygons = []

            if (self.df[self.df['id'] == image_id]['dataset'] == 1).bool():
                subset = 'train'
            else:
                subset = 'val'

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
                dsitems.append(
                    DatasetItem(
                        id=index,
                        annotations=datumaro_polygons,
                        image=img,
                        attributes=attributes,
                        subset=subset))

        dataset = Dataset.from_iterable(dsitems, categories=self.labels)
        dataset.export(f"{export_path}", 'coco', default_image_ext='.tif', save_media=True)


if __name__ == '__main__':
    dataset = HuBMAPVasculatureDataset(
        data_root='/home/yuchunli/_DATASET/hubmap-hacking-the-human-vasculature')

    dataset.make_coco_train(export_path="/home/yuchunli/_DATASET/HuBMAP-vasculature-coco")
