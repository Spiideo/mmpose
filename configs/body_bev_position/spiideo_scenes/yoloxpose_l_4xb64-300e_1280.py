from copy import deepcopy

_base_ = 'yoloxpose_l_4xb64-300e_640.py'

xside = yside = 1280
input_size = (xside, yside)
max_input_size = (int(1.25*xside), int(1.25*yside))

model = dict(data_preprocessor=dict(batch_augments=[dict(
    type='BatchSyncRandomResize',
    random_size_range=(int(0.75 * xside), int(1.25 * yside)),
    size_divisor=32,
    interval=1,
)]))
codec = dict(input_size=max_input_size)

train_pipeline_stage1 = deepcopy(_base_.train_pipeline_stage1)
train_pipeline_stage1[1]['img_scale'] = max_input_size
train_pipeline_stage1[2]['input_size'] = max_input_size
train_pipeline_stage1[3]['img_scale'] = max_input_size
train_dataloader = dict(dataset=dict(pipeline=train_pipeline_stage1))

train_pipeline_stage2 = deepcopy(_base_.train_pipeline_stage2)
train_pipeline_stage2[1]['input_size'] = max_input_size
custom_hooks = deepcopy(_base_.custom_hooks)
custom_hooks[0]['new_train_pipeline'] = train_pipeline_stage2

val_pipeline = deepcopy(_base_.val_pipeline)
val_pipeline[1]['input_size'] = input_size
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = val_dataloader

visualizer = deepcopy(_base_.visualizer)
visualizer['vis_backends'][0]['init_kwargs']['task_name'] = f'yoloxpose_l_8xb32-300e_coco-{input_size}'
