_base_ = [
    '../../body_2d_keypoint/yoloxpose/coco/yoloxpose_tiny_4xb64-300e_coco-416.py',
    'soccer_v1.py',
]

train_dataloader = dict(dataset=_base_.train_dataset)
val_dataloader = dict(dataset=_base_.val_dataset)
test_dataloader = dict(dataset=_base_.test_dataset)
val_evaluator = dict(ann_file=_base_.val_evaluator_ann_file, iou_type='bbox', use_area=False)
test_evaluator = dict(ann_file=_base_.test_evaluator_ann_file, iou_type='bbox', use_area=False)

model = dict(head=dict(
    num_keypoints=2,
    assigner=dict(oks_calculator=dict(type='PoseOKS', metainfo=_base_.dataset_metainfo)),
    loss_oks=dict(metainfo=_base_.dataset_metainfo),
))
