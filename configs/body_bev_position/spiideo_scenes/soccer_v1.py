
dataset_type = 'SpiideoScenes'
dataset_root = 'data/SpiideoScenes/Soccer/v1/'

train_dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train'),
)
val_dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='val'),
)
test_dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test'),
)

val_evaluator_ann_file = dataset_root + 'annotations/val.json'
test_evaluator_ann_file = dataset_root + 'annotations/test.json'