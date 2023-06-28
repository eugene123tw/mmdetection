import os

import cv2
import numpy as np
import pandas as pd

data_root = '/home/yuchunli/_DATASET/hubmap-hacking-the-human-vasculature/train'
df = pd.read_csv(
    '/home/yuchunli/_DATASET/hubmap-hacking-the-human-vasculature/tile_meta.csv'
)

# df = df.drop(df[df.dataset == 3].index)

for source_wsi in df['source_wsi'].unique():
    new_df = df[df['source_wsi'] == source_wsi].reset_index(drop=True)

    left = new_df['i'].min()
    top = new_df['j'].min()
    right = new_df['i'].max() + 512
    bottom = new_df['j'].max() + 512
    img_w = right - left
    img_h = bottom - top
    canvas = np.full((img_h // 4, img_w // 4, 3), 0, np.uint8)
    for _, row in new_df.iterrows():
        img = cv2.imread(os.path.join(data_root, f'{row["id"]}.tif'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512 // 4, 512 // 4))
        x1 = (row['i'] - left) // 4
        y1 = (row['j'] - top) // 4
        x2 = x1 + 512 // 4
        y2 = y1 + 512 // 4
        if np.sum(canvas[y1:y2, x1:x2, :]) != 0:
            print(f"Overlap: {row['i']}, {row['j']}")
        canvas[y1:y2, x1:x2, :] = img
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    cv2.imwrite(
        f'/home/yuchunli/git/mmdetection/demo/hubmap/wsi_{source_wsi}.png',
        canvas)

for slide_id, df_slide in df.groupby('source_wsi'):
    left = df_slide['i'].min()
    top = df_slide['j'].min()
    right = df_slide['i'].max() + 512
    bottom = df_slide['j'].max() + 512
    img_w = right - left
    img_h = bottom - top
    canvas = np.full((img_h, img_w, 3), 0, np.uint8)
    for _, row in df_slide.iterrows():
        img = cv2.imread(os.path.join(data_root, f'{row["id"]}.tif'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x1 = row['i'] - left
        y1 = row['j'] - top
        x2 = x1 + 512
        y2 = y1 + 512
        # if np.sum(canvas) != 0:
        # print(f"Overlap: {slide_id}, {row['i']}, {row['j']}")
        canvas[y1:y2, x1:x2, :] = img
    print()
