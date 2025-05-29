import json
from torch.utils.tensorboard import SummaryWriter

def json_logs_to_tensorboard(log_file, tb_logdir):
    writer = SummaryWriter(tb_logdir)

    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            log = json.loads(line)

            step = log.get('step', None)

            # 训练日志: 有 'epoch' 字段，包含loss和acc
            if 'epoch' in log:
                # 写入训练loss及准确率
                for key in ['loss', 'loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox', 'loss_mask', 'acc', 'lr']:
                    if key in log:
                        writer.add_scalar(f'train/{key}', float(log[key]), step)

            # 验证日志: 没有 'epoch' 字段，包含 mAP 指标
            else:
                # 写入验证 mAP 指标
                for key, val in log.items():
                    if 'mAP' in key or 'precision' in key or 'recall' in key:
                        tb_key = key.replace('/', '_')  # 替换斜杠避免TensorBoard层级混乱
                        writer.add_scalar(f'val/{tb_key}', float(val), step)

                # 如果验证日志里有loss，也写入（可选）
                for key in ['loss', 'loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox', 'loss_mask']:
                    if key in log:
                        writer.add_scalar(f'val/{key}', float(log[key]), step)

    writer.close()
    print(f'TensorBoard日志写入完成，路径: {tb_logdir}')

if __name__ == '__main__':
    log_file = 'mmdetection/work_dirs/sparse-rcnn_r50_fpn_1x_voc0712/20250528_165836/vis_data/20250528_165836.json'    # 替换成你的日志文件名
    tb_logdir = 'mmdetection/work_dirs/sparse-rcnn_r50_fpn_1x_voc0712/20250528_165836/tensorboard_logs'    # TensorBoard日志存放目录
    json_logs_to_tensorboard(log_file, tb_logdir)
