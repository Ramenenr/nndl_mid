import os
import numpy as np
import torch
import mmcv
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS

register_all_modules()
init_default_scope('mmdet')

# 配置与模型路径
mask_config = 'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_voc.py'
mask_checkpoint = 'work_dirs/mask_rcnn_r50_fpn_voc/epoch_12.pth'

# 加载模型
mask_model = init_detector(mask_config, mask_checkpoint, device='cuda:0')

# 图像路径
img_dir = 'data/VOCdevkit/VOC2012/JPEGImages'
test_images = [
    '2008_000008.jpg',
    '2008_000064.jpg',
    '2008_000084.jpg'
]

def get_img_meta(img):
    return [{
        'img_shape': (img.shape[0], img.shape[1], 3),
        'scale_factor': np.array([1., 1., 1., 1.], dtype=np.float32),
        'ori_shape': (img.shape[0], img.shape[1], 3)
    }]

def extract_rpn_proposals(model, img_tensor, img_meta):
    with torch.no_grad():
        feat = model.extract_feat(img_tensor)
        rpn_outs = model.rpn_head(feat)
        proposal_cfg = model.test_cfg.rpn
        proposals = model.rpn_head.predict_by_feat(*rpn_outs, batch_img_metas=img_meta, cfg=proposal_cfg)
    return proposals

def show_bboxes(img, bboxes, title='RPN Proposals', top_k=100):
    img_rgb = mmcv.bgr2rgb(img.copy())
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    for box in bboxes[:top_k]:
        x1, y1, x2, y2 = box
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                          fill=False, edgecolor='lime', linewidth=1.5))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_prediction(model, img, title='Prediction', score_thr=0.3):
    result = inference_detector(model, img)
    num_instances = 0
    if isinstance(result, (list, tuple)):
        for res in result:
            if hasattr(res, 'pred_instances'):
                num_instances += len(res.pred_instances)
            elif isinstance(res, dict) and 'scores' in res:
                num_instances += len(res['scores'])
    print(f"🔍 {title} - Detected {num_instances} instances")

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    os.makedirs('outputs', exist_ok=True)
    visualizer.add_datasample(
        name=title,
        image=img,
        data_sample=result,
        draw_gt=False,
        show=False,
        out_file=f'outputs/{title.replace(" ", "_")}.jpg',
        pred_score_thr=score_thr
    )

# 推理流程
for img_name in test_images:
    print(f"\n🖼️ Processing image: {img_name}")
    img_path = os.path.join(img_dir, img_name)
    img = mmcv.imread(img_path)

    # RPN proposals
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
    img_meta = get_img_meta(img)
    proposals = extract_rpn_proposals(mask_model, img_tensor, img_meta)
    rpn_boxes = proposals[0].bboxes.cpu().numpy()
    show_bboxes(img, rpn_boxes, title=f'RPN Proposals - {img_name}', top_k=100)

    # Prediction + 可视化
    show_prediction(mask_model, img, title=f'Mask R-CNN Prediction - {img_name}', score_thr=0.1)
