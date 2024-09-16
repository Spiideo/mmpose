_base_ = './yoloxpose_s_8xb32-300e_coco-640.py'

widen_factor = 0.375
deepen_factor = 0.33
checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_' \
    'tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'

# model settings
model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        init_cfg=dict(checkpoint=checkpoint),
    ),
    neck=dict(in_channels=[96, 192, 384], out_channels=96,),
    head=dict(head_module_cfg=dict(widen_factor=widen_factor), ))
