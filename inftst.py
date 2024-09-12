from mmpose.apis import MMPoseInferencer

# img_path = '000007.jpg'
img_path = '00000001.jpg'

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer(
    pose2d='configs/body_bev_position/spiideo_scenes/yoloxpose_tiny_4xb64-300e_416.py',
    pose2d_weights='epoch_170.pth')

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer(img_path, show=True)
result = next(result_generator)
