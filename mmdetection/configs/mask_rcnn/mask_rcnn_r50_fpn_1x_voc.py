_base_ = 'mask-rcnn_r50_fpn_1x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20),
        mask_head=dict(num_classes=20)
    )
)

dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'

metainfo = dict(
    classes=(
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
        'dog', 'horse', 'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor'
    )
)

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2007/ImageSets/Segmentation/train.txt',
        data_prefix=dict(img_path='VOC2007/JPEGImages', seg_map_path='VOC2007/SegmentationClass'),
        metainfo=metainfo
    )
)

val_dataloader = train_dataloader
test_dataloader = train_dataloader

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend', save_dir='vis_outputs'),
        dict(type='TensorboardVisBackend', save_dir='vis_outputs/tensorboard')
    ],
    name='visualizer'
)
