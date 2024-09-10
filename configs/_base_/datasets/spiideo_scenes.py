dataset_info = dict(
    dataset_name='spiideo_scene',
    # paper_info=dict(
    #     author='Mykhaylo Andriluka and Leonid Pishchulin and '
    #     'Peter Gehler and Schiele, Bernt',
    #     title='2D Human Pose Estimation: New Benchmark and '
    #     'State of the Art Analysis',
    #     container='IEEE Conference on Computer Vision and '
    #     'Pattern Recognition (CVPR)',
    #     year='2014',
    #     homepage='http://human-pose.mpi-inf.mpg.de/',
    # ),
    keypoint_info={
        0:
        dict(
            name='pelvis',
            id=0,
            color=[255, 0, 0],
            type='lower',
            swap='pelvis'),
        1:
        dict(
            name='pelvis_ground',
            id=0,
            color=[0, 255, 0],
            type='lower',
            swap='pelvis_ground'),
    },
    skeleton_info={
        0:
        dict(link=('pelvis', 'pelvis_ground'), id=0, color=[255, 255, 0]),
    },
    joint_weights=[
        1., 1.
    ],
    sigmas=[
        0.089, 0.089,
    ])