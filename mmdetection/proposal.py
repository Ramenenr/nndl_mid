from mmdet.registry import VISUALIZERS
import mmcv
import os
import matplotlib.pyplot as plt
from PIL import Image
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

# 初始化
register_all_modules()

# 配置和权重路径（只加载 Mask R-CNN）
config_mask_rcnn = 'configs/pascal_voc/mask-rcnn_r50_fpn_1x_voc0712.py'
checkpoint_mask_rcnn = 'work_dirs/mask-rcnn_r50_fpn_1x_voc0712/epoch_12.pth'
model_mask_rcnn = init_detector(config_mask_rcnn, checkpoint_mask_rcnn, device='cuda:0')

# 判断是否为 notebook
def is_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell in ['ZMQInteractiveShell', 'Shell']
    except:
        return False

# 可视化 proposal box（不画最终预测）
def visualize_mask_rcnn_proposals_only(model, image_path, out_dir='vis/mask_rcnn_proposals', image_name='demo.jpg'):
    os.makedirs(out_dir, exist_ok=True)
    image = mmcv.imread(image_path, channel_order='rgb')
    result = inference_detector(model, image)

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    out_file = os.path.join(out_dir, f'{image_name}_proposals.jpg')
    visualizer.add_datasample(
        name='only_proposals',
        image=image,
        data_sample=result,
        draw_gt=False,
        draw_pred=True,         # ✅ 保留预测框以触发绘图逻辑
        pred_score_thr=0.0,     # ✅ 显示所有 proposal（含低分）
        show=False,
        out_file=out_file
    )

    # 显示图片（可选）
    if is_notebook():
        display(Image.open(out_file))
    else:
        img = mmcv.imread(out_file, channel_order='rgb')
        plt.imshow(img)
        plt.axis('off')
        plt.title('Proposal Boxes (Mask R-CNN)')
        plt.show()

# 批量处理 VOC 测试集前 10 张图片
with open('data/VOCdevkit/VOC2007/ImageSets/Segmentation/test.txt', 'r') as f:
    test_ids = [line.strip() for line in f.readlines()]

for image_id in test_ids[:10]:
    image_path = f'data/VOCdevkit/VOC2007/JPEGImages/{image_id}.jpg'
    visualize_mask_rcnn_proposals_only(model_mask_rcnn, image_path,
                                       out_dir='vis/mask_rcnn_proposals',
                                       image_name=image_id)
