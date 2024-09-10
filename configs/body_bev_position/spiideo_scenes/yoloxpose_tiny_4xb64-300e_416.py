_base_ = [
    '../../body_2d_keypoint/yoloxpose/coco/yoloxpose_tiny_4xb64-300e_coco-416.py',
    'soccer_v1.py',
]

train_dataloader = dict(dataset=_base_.train_dataset)
val_dataloader = dict(dataset=_base_.val_dataset)
test_dataloader = dict(dataset=_base_.test_dataset)
val_evaluator = dict(ann_file=_base_.val_evaluator_ann_file)
test_evaluator = dict(ann_file=_base_.test_evaluator_ann_file)
