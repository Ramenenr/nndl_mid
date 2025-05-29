from mmdet.registry import VISUALIZERS
import mmcv
import os
import matplotlib.pyplot as plt
from PIL import Image
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

# 初始化
config_file = 'configs/pascal_voc/mask-rcnn_r50_fpn_1x_voc0712.py'
checkpoint_file = 'work_dirs/mask-rcnn_r50_fpn_1x_voc0712/epoch_12.pth'
register_all_modules()
model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or 'cpu'

# 判断是否在 Jupyter Notebook
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        return shell in ['ZMQInteractiveShell', 'Shell']
    except NameError:
        return False

# 可视化函数
def visualize_mask_rcnn_proposal_vs_final(model, image_path, out_dir='vis/mask_rcnn_compare', image_name='demo.jpg'):
    os.makedirs(out_dir, exist_ok=True)
    image = mmcv.imread(image_path, channel_order='rgb')
    result = inference_detector(model, image)

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # 可视化 Proposal Boxes
    proposal_out = os.path.join(out_dir, f'{image_name}_proposal.jpg')
    visualizer.add_datasample(
        name='proposals',
        image=image,
        data_sample=result,
        draw_gt=False,
        draw_pred=True,
        draw_proposal=True,      # ✅ 显示 proposal
        pred_score_thr=0.0,      # 显示全部 proposal，不进行筛选
        show=False,
        out_file=proposal_out
    )

    # 可视化 Final Prediction
    final_out = os.path.join(out_dir, f'{image_name}_final.jpg')
    visualizer.add_datasample(
        name='final_result',
        image=image,
        data_sample=result,
        draw_gt=False,
        draw_pred=True,          # ✅ 显示最终预测框
        draw_proposal=False,     # ❌ 不显示 proposal
        pred_score_thr=0.3,      # 筛掉低分预测框
        show=False,
        out_file=final_out
    )

    # 显示图片（如非 Notebook 可选）
    if is_notebook():
        display(Image.open(proposal_out))
        display(Image.open(final_out))
    else:
        for f, title in [(proposal_out, 'Proposals'), (final_out, 'Final Prediction')]:
            img = mmcv.imread(f, channel_order='rgb')
            plt.imshow(img)
            plt.axis('off')
            plt.title(title)
            plt.show()

# 示例批量处理 VOC 测试集前10张图像
with open('data/VOCdevkit/VOC2007/ImageSets/Segmentation/test.txt', 'r') as f:
    image_list = [line.strip() for line in f.readlines()]

for image_idx in image_list[:10]:
    image_path = f'data/VOCdevkit/VOC2007/JPEGImages/{image_idx}.jpg'
    visualize_mask_rcnn_proposal_vs_final(model, image_path, out_dir='vis/mask_rcnn_compare', image_name=image_idx)
