_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

# 1. 修改类别
num_classes = 20
classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
        mask_head=dict(num_classes=num_classes),
    )
)

# 2. 数据集设置
dataset_type = 'CocoDataset'
data_root = 'data/VOCdevkit/'

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='coco/voc07_trainval.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=_base_.train_pipeline,
        metainfo=dict(classes=classes),
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='coco/voc07_test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=_base_.test_pipeline,
        metainfo=dict(classes=classes),
    )
)

test_dataloader = val_dataloader

# 3. 评估器设置
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'coco/voc07_test.json',
    metric=['bbox', 'segm'],
    format_only=False,
    classwise=True,
)
test_evaluator = val_evaluator

# 4. 加载 backbone 预训练模型（不会加载 bbox/mask head）
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/' \
            'mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
