_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1x.py'
]

# VOC 类别定义
dataset_type = 'CocoDataset'
data_root = 'data/VOCdevkit/VOC2012/'
classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]
metainfo = dict(classes=classes)

# 数据处理 pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs')
]

# DataLoader 设置
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'voc2012_train_coco.json',
        data_prefix=dict(img=data_root + 'JPEGImages/'),
        metainfo=metainfo,
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'voc2012_val_coco.json',
        data_prefix=dict(img=data_root + 'JPEGImages/'),
        metainfo=metainfo,
        pipeline=test_pipeline
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'voc2012_test_coco.json',
        data_prefix=dict(img=data_root + 'JPEGImages/'),
        metainfo=metainfo,
        pipeline=test_pipeline
    )
)

# 评估器
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'voc2012_val_coco.json',
    metric=['bbox', 'segm']
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'voc2012_test_coco.json',
    metric=['bbox', 'segm']
)

# 模型设置（修改类别数）
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20),
        mask_head=dict(num_classes=20)
    )
)

# 优化器设置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# 训练流程控制
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=1
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 日志记录与可视化
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# TensorBoard 可视化支持
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# 环境配置
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

# 模型加载路径（使用 COCO 预训练模型）
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/' \
            'mask_rcnn_r50_fpn_1x_coco/' \
            'mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
resume = False

# 训练默认作用域
default_scope = 'mmdet'
log_level = 'INFO'
