import PIL
from datasets import load_dataset
from datumaro import AnnotationType, DatasetItem, Bbox, LabelCategories
from datumaro.components.project import Dataset, Environment
from datumaro.util.image import Image
import numpy as np
from sklearn.model_selection import train_test_split

PIL.Image.MAX_IMAGE_PIXELS = 933120000

data_root = "/home/yuchunli/git/ship-detection"

data = load_dataset(data_root)

train_indices, val_indices = train_test_split(
    range(len(data['train'])), test_size=0.2, random_state=42)

categories={
    AnnotationType.label: LabelCategories.from_iterable(['ship', 'dog'])
}

dsitems = []
for index, dataset_item in enumerate(data['train']):
    attributes = {'filename': dataset_item['image'].filename}
    subset = "train" if index in train_indices else "val"
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
                    subset=subset,
                    image=image,
                    attributes=attributes)
    )

dataset = Dataset.from_iterable(dsitems, categories=categories)
dataset.export('/home/yuchunli/git/ship-detection/coco', 'coco', save_images=True)