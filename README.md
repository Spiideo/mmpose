# Spiideo SoccerNet SynLoc - Single Frame World Coordinate Athlete Detection and Localization with Synthetic Data

This is the official code release for baseline method presented in the Spiideo SoccerNet SynLoc paper. It is
based on [mmpose](https://github.com/open-mmlab/mmpose).

## Installation

## Pre-trained models

| Model | Input Size | GFLOPS | mAP-LocSim | Precision | Recall | F1 | Frame Acc. | Download |
|:-----:|:----------:|:------:|:----------:|:---------:|:------:|:--:|:----------:|:--------:|
| [YOLOX-tiny](configs/body_bev_position/spiideo_scenes/yoloxpose_tiny_4xb64-300e_640.py)| 640 | 10.3 | 60.9 | 83.6 | 75.0 | 79.1 | 9.4 | [model]() \| [log]()
| [YOLOX-s](configs/body_bev_position/spiideo_scenes/yoloxpose_s_4xb64-300e_640.py)| 640 | 18.3 | 64.4 | 85.4 | 78.0 | 81.5 | 11.6 | [model]() \| [log]()
| [YOLOX-m](configs/body_bev_position/spiideo_scenes/yoloxpose_m_4xb64-300e_640.py)| 640 | 47.9 | 68.2 | 87.8 | 81.0 | 84.3 | 16.6 | [model]() \| [log]()
| [YOLOX-tiny](configs/body_bev_position/spiideo_scenes/yoloxpose_tiny_4xb64-300e_960.py)| 960 | 23.3 | 73.3 | 88.1 | 86.0 | 87.0 | 23.1 | [model]() \| [log]()
| [YOLOX-s](configs/body_bev_position/spiideo_scenes/yoloxpose_s_4xb64-300e_960.py)| 960 | 41.1 | 76.6 | 91.4 | 88.0 | 89.7 | 27.2 | [model]() \| [log]()
| [YOLOX-m](configs/body_bev_position/spiideo_scenes/yoloxpose_m_4xb64-300e_960.py)| 960 | 108.0 | 79.7 | 93.1 | 89.0 | 91.0 | 30.7 | [model]() \| [log]()

## Evaluation

## Training

## Citation

If you use this code for your research, please cite:

