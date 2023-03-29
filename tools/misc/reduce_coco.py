# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import argparse
import json
import os


def shorten_annotation(src_path, dst_path, num_images):
    """Shorten annotation JSON file to contain only first num_images.

    Args:
        src_path (_type_): _description_
        dst_path (_type_): _description_
        num_images (int): number of images to keep
    """
    with open(src_path) as read_file:
        content = json.load(read_file)
        selected_indexes = sorted([item['id'] for item in content['images']])
        selected_indexes = selected_indexes[:num_images]
        content['images'] = [
            item for item in content['images']
            if item['id'] in selected_indexes
        ]
        content['annotations'] = [
            item for item in content['annotations']
            if item['image_id'] in selected_indexes
        ]
        content['licenses'] = [
            item for item in content['licenses']
            if item['id'] in selected_indexes
        ]

    with open(dst_path, 'w') as write_file:
        json.dump(content, write_file)


def parse_args():
    parser = argparse.ArgumentParser(
        'Reduce number of samples in annotation JSON file')
    parser.add_argument(
        '--input',
        type=str,
        help='input annotation JSON i.e. annotation_train.json')
    parser.add_argument(
        '--num', type=int, required=True, help='Number of samples to keep')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert os.path.exists(args.input) and args.input.endswith('.json')
    output_dir = os.path.dirname(args.input)
    basename = os.path.splitext(os.path.basename(args.input))[0]
    output_basename = f'{basename}_shorten_to_{args.num}.json'
    output_path = os.path.join(output_dir, output_basename)
    shorten_annotation(args.input, output_path, args.num)


if __name__ == '__main__':
    main()
