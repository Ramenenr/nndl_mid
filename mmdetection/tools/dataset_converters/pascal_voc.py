# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import xml.etree.ElementTree as ET

import numpy as np
from mmengine.fileio import dump, list_from_file
from mmengine.utils import mkdir_or_exist, track_progress

from mmdet.evaluation import voc_classes

label_ids = {name: i for i, name in enumerate(voc_classes())}


def parse_xml(args):
    xml_path, img_path = args
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    bboxes = []
    labels = []
    bboxes_ignore = []
    labels_ignore = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        label = label_ids[name]
        difficult = int(obj.find('difficult').text)
        bnd_box = obj.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        if difficult:
            bboxes_ignore.append(bbox)
            labels_ignore.append(label)
        else:
            bboxes.append(bbox)
            labels.append(label)
    if not bboxes:
        bboxes = np.zeros((0, 4))
        labels = np.zeros((0, ))
    else:
        bboxes = np.array(bboxes, ndmin=2) - 1
        labels = np.array(labels)
    if not bboxes_ignore:
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0, ))
    else:
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
        labels_ignore = np.array(labels_ignore)
    annotation = {
        'filename': img_path,
        'width': w,
        'height': h,
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
            'bboxes_ignore': bboxes_ignore.astype(np.float32),
            'labels_ignore': labels_ignore.astype(np.int64)
        }
    }
    return annotation


def cvt_annotations(devkit_path, years, split, out_file):
    if not isinstance(years, list):
        years = [years]
    annotations = []
    for year in years:
        filelist = osp.join(devkit_path,
                            f'VOC{year}/ImageSets/Main/{split}.txt')
        if not osp.isfile(filelist):
            print(f'filelist does not exist: {filelist}, '
                  f'skip voc{year} {split}')
            return
        img_names = list_from_file(filelist)
        xml_paths = [
            osp.join(devkit_path, f'VOC{year}/Annotations/{img_name}.xml')
            for img_name in img_names
        ]
        img_paths = [
            f'VOC{year}/JPEGImages/{img_name}.jpg' for img_name in img_names
        ]
        part_annotations = track_progress(parse_xml,
                                          list(zip(xml_paths, img_paths)))
        annotations.extend(part_annotations)
    if out_file.endswith('json'):
        annotations = cvt_to_coco_json(annotations)
    dump(annotations, out_file)
    return annotations


def cvt_to_coco_json(annotations):
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []

    # category_id 统一使用与 label_ids 对应的值（0-based → 1-based）
    category_id_map = {v: i + 1 for i, (k, v) in enumerate(label_ids.items())}
    for name, i in label_ids.items():
        category_item = {
            'supercategory': 'none',
            'id': category_id_map[i],  # 保证从1开始
            'name': name
        }
        coco['categories'].append(category_item)

    image_set = set()
    annotation_id = 0
    image_id = 0

    for ann_dict in annotations:
        file_name = ann_dict['filename']
        ann = ann_dict['ann']
        assert file_name not in image_set
        image_item = {
            'id': image_id,
            'file_name': file_name,
            'height': ann_dict['height'],
            'width': ann_dict['width']
        }
        coco['images'].append(image_item)
        image_set.add(file_name)

        bboxes = ann['bboxes'][:, :4]
        labels = ann['labels']
        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            label = category_id_map[int(labels[bbox_id])]  # 转成 coco 的分类 id（从1开始）
            annotation_item = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': label,
                'bbox': [
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2] - bbox[0]),
                    float(bbox[3] - bbox[1])
                ],
                'area': float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
                'iscrowd': 0,
                'ignore': 0,
                'segmentation': [[
                    float(bbox[0]), float(bbox[1]),
                    float(bbox[0]), float(bbox[3]),
                    float(bbox[2]), float(bbox[3]),
                    float(bbox[2]), float(bbox[1])
                ]]
            }
            coco['annotations'].append(annotation_item)
            annotation_id += 1

        bboxes_ignore = ann['bboxes_ignore'][:, :4]
        labels_ignore = ann['labels_ignore']
        for bbox_id in range(len(bboxes_ignore)):
            bbox = bboxes_ignore[bbox_id]
            label = category_id_map[int(labels_ignore[bbox_id])]
            annotation_item = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': label,
                'bbox': [
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2] - bbox[0]),
                    float(bbox[3] - bbox[1])
                ],
                'area': float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
                'iscrowd': 1,
                'ignore': 0,
                'segmentation': [[
                    float(bbox[0]), float(bbox[1]),
                    float(bbox[0]), float(bbox[3]),
                    float(bbox[2]), float(bbox[3]),
                    float(bbox[2]), float(bbox[1])
                ]]
            }
            coco['annotations'].append(annotation_item)
            annotation_id += 1

        image_id += 1

    return coco


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PASCAL VOC annotations to mmdetection format')
    parser.add_argument('devkit_path', help='pascal voc devkit path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--out-format',
        default='pkl',
        choices=('pkl', 'coco'),
        help='output format, "coco" indicates coco annotation format')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    devkit_path = args.devkit_path
    out_dir = args.out_dir if args.out_dir else devkit_path
    mkdir_or_exist(out_dir)

    years = []
    if osp.isdir(osp.join(devkit_path, 'VOC2007')):
        years.append('2007')
    if osp.isdir(osp.join(devkit_path, 'VOC2012')):
        years.append('2012')
    if '2007' in years and '2012' in years:
        years.append(['2007', '2012'])
    if not years:
        raise IOError(f'The devkit path {devkit_path} contains neither '
                      '"VOC2007" nor "VOC2012" subfolder')
    out_fmt = f'.{args.out_format}'
    if args.out_format == 'coco':
        out_fmt = '.json'
    for year in years:
        if year == '2007':
            prefix = 'voc07'
        elif year == '2012':
            prefix = 'voc12'
        elif year == ['2007', '2012']:
            prefix = 'voc0712'
        for split in ['train', 'val', 'trainval']:
            dataset_name = prefix + '_' + split
            print(f'processing {dataset_name} ...')
            cvt_annotations(devkit_path, year, split,
                            osp.join(out_dir, dataset_name + out_fmt))
        if not isinstance(year, list):
            dataset_name = prefix + '_test'
            print(f'processing {dataset_name} ...')
            cvt_annotations(devkit_path, year, 'test',
                            osp.join(out_dir, dataset_name + out_fmt))
    print('Done!')


if __name__ == '__main__':
    main()
