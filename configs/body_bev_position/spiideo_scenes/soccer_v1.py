
dataset_type = 'SpiideoScenes'
dataset_root = 'data/SpiideoScenes/Soccer/v1/'
dataset_metainfo = 'configs/_base_/datasets/spiideo_scenes.py'

train_dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train'),
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
)
val_dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='val'),
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
)
test_dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test'),
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
)

val_evaluator_ann_file = dataset_root + 'annotations/val.json'
test_evaluator_ann_file = dataset_root + 'annotations/test.json'
