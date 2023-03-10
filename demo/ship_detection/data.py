import PIL
from datasets import load_dataset
from datumaro import AnnotationType, DatasetItem, Bbox, LabelCategories
from datumaro.components.project import Dataset, Environment
from datumaro.util.image import Image
import numpy as np
from sklearn.model_selection import train_test_split
import math

PIL.Image.MAX_IMAGE_PIXELS = 933120000


class ShipDetectionDataset:
    """ Ship Detection from HuggingFace Dataset
    """
    def __init__(self, data_root) -> None:
        self.dataset = load_dataset(data_root)
        self.categories = {
            AnnotationType.label: LabelCategories.from_iterable(['ship'])}

    def make_coco(self, export_path):
        train_indices, val_indices = train_test_split(
            np.arange(len(self.dataset['train'])),
            test_size=0.2,
            random_state=42)

        dsitems = []
        for subset in ['train', 'test']:
            for index, dataset_item in enumerate(self.dataset[subset]):
                attributes = {'filename': dataset_item['image'].filename}
                if subset == 'train':
                    subset_str = "train" if index in train_indices else "val"
                else:
                    subset_str = "test"
                np_image = np.array(dataset_item['image'])
                image = Image(data=np_image, size=np_image.shape[0:2])
                bboxes = []
                for bbox, label in zip(dataset_item['objects']['bbox'],
                                       dataset_item['objects']['categories']):
                    if label != 0:
                        raise RuntimeError("Label is not 0")
                    x1, y1, x2, y2 = bbox
                    box = Bbox(
                        x=x1,
                        y=y1,
                        w=(x2 - x1),
                        h=(y2 - y1),
                        label=label
                    )
                    bboxes.append(box)

                dsitems.append(
                    DatasetItem(id=index,
                                annotations=bboxes,
                                subset=subset_str,
                                image=image,
                                attributes=attributes)
                )

        dataset = Dataset.from_iterable(dsitems, categories=self.categories)
        dataset.export(export_path, 'coco', save_images=True)

    def get_adaptive_tile_params(self, object_tile_ratio=0.01, rule="avg"):
        tile_cfg = dict(tile_size=None, tile_overlap=None, tile_max_number=None)
        bboxes = np.zeros((0, 4), dtype=np.float32)
        image_sizes = []
        max_object = 0
        for dataset_item in self.dataset['train']:
            image_sizes.extend([dataset_item['image'].size] * len(dataset_item['objects']['bbox']))
            anno = dataset_item['objects']
            bbox = np.array(anno['bbox'])
            bboxes = np.concatenate((bboxes, bbox), 0)
            if len(bbox) > max_object:
                max_object = len(bbox)

        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

        if rule == "min":
            object_area = np.min(areas)
        elif rule == "avg":
            object_area = np.median(areas)

        # NOTE: another strategy is to compute the minimum detectable object size
        # For instance, in order to have > 1 pixel in the smallest feature map, the minimum object size should be VFNet the largest stride is 128

        max_area = np.max(areas)

        tile_size = int(math.sqrt(object_area/object_tile_ratio))
        overlap_ratio = max_area/(tile_size**2) if max_area/(tile_size**2) < 1.0 else None

        tile_cfg.update(dict(tile_size=tile_size, tile_max_number=max_object))
        if overlap_ratio:
            tile_cfg.update(dict(tile_overlap=overlap_ratio))
        print(tile_cfg)
        return tile_cfg


if __name__ == '__main__':
    dataset = ShipDetectionDataset(data_root="/home/yuchunli/_DATASET/ship-detection")
    dataset.get_adaptive_tile_params(rule='min')
    # dataset.make_coco(export_path="/home/yuchunli/_DATASET/ship-detection-coco")