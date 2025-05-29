# 1.环境配置

### 安装mmdetection

```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
pip3 install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
mim install mmdet
git clone https://github.com/open-mmlab/mmdetection
wget https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
```

# 2.数据准备

### 下载VOC2007数据集

```bash
cd mmdetection
# 创建数据集目录
mkdir -p data
cd data

# 下载VOC2007数据集
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

# 解压数据集
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
```

### 将VOC数据转化为COCO数据格式

```bash
# 在mmdetection目录下运行
python tools/dataset_converters/pascal_voc.py \
    data/VOCdevkit \
    --out-dir data/VOCdevkit/coco \
    --out-format coco
```

# 3.模型训练

### Mask R-CNN训练

```bash
# 在mmdetection目录下运行
bash tools/dist_train.sh configs/pascal_voc/mask-rcnn_r50_fpn_1x_voc0712.py 8
```

### Sparse R-CNN训练

```bash
# 在mmdetection目录下运行
bash tools/dist_train.sh configs/pascal_voc/mask-rcnn_r50_fpn_1x_voc0712.py 8
```

# 4.推理

挑选测试集中的图像，通过可视化对比训练好的Mask R-CNN第一阶段产生的proposal box和最终的预测结果

```bash
# 在mmdetection目录下运行
python proposal.py
```

用测试集中的图像，和三张不在VOC数据集内包含有VOC中类别物体的图像可视化Mask R-CNN 和Sparse R-CNN的实例分割与目标检测结果

```bash
python infer.py
```

# 5.训练log可视化

将训练log转化为tensorboard能够处理的形式。对于**Mask R-CNN**：

```bash
python parse_mmdet_log_to_tensorboard_mask.py \
    mmdetection/work_dirs/mask-rcnn_r50_fpn_1x_voc0712/20250528_154515/vis_data/20250528_154515.json \
    --tb_log_dir mmdetection/work_dirs/mask-rcnn_r50_fpn_1x_voc0712/20250528_154515/tensorboard_logs
```

```bash
tensorboard --logdir mmdetection/work_dirs/mask-rcnn_r50_fpn_1x_voc0712/20250528_154515/tensorboard_logs
```

对于**Sparse R-CNN**：

```bash
python parse_mmdet_log_to_tensorboard.py \
    mmdetection/work_dirs/sparse-rcnn_r50_fpn_1x_voc0712/20250528_165836/vis_data/20250528_165836.json \
    --tb_log_dir mmdetection/work_dirs/sparse-rcnn_r50_fpn_1x_voc0712/20250528_165836/tensorboard_logs
```

```bash
tensorboard --logdir mmdetection/work_dirs/mask-rcnn_r50_fpn_1x_voc0712/20250528_165836/tensorboard_logs
```




​    
