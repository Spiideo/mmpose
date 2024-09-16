_base_ = [
    '../../body_2d_keypoint/yoloxpose/coco/yoloxpose_m_8xb32-300e_coco-640.py',
    'soccer_v1.py',
]

train_dataloader = dict(dataset=_base_.train_dataset, num_workers=16, batch_size=64)
val_dataloader = dict(dataset=_base_.val_dataset, num_workers=4)
test_dataloader = dict(dataset=_base_.test_dataset, num_workers=4)
val_evaluator = _base_.bev_val_evaluator
test_evaluator = _base_.bev_test_evaluator
visualizer = _base_.clearml_visualizer

model = dict(head=dict(
    num_keypoints=2,
    assigner=dict(oks_calculator=dict(type='PoseOKS', metainfo=_base_.dataset_metainfo)),
    loss_oks=dict(metainfo=_base_.dataset_metainfo),
))
