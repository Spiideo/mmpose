# Spiideo SoccerNet SynLoc - Single Frame World Coordinate Athlete Detection and Localization with Synthetic Data

This is the official code release for baseline method presented in the Spiideo SoccerNet SynLoc paper. It is
a fork of the [mmpose](https://github.com/open-mmlab/mmpose) repo.

## Installation

## Pre-trained models

| Model | Input Size | GFLOPS | mAP-LocSim | Precision | Recall | F1 | Frame Acc. | Download |
|:-----:|:----------:|:------:|:----------:|:---------:|:------:|:--:|:----------:|:--------:|
| [YOLOX-tiny](configs/body_bev_position/spiideo_scenes/yoloxpose_tiny_4xb64-300e_640.py)| 640 | 10.3 | 60.9 | 83.6 | 75.0 | 79.1 | 9.4 | [model](),  [log]()
| [YOLOX-s](configs/body_bev_position/spiideo_scenes/yoloxpose_s_4xb64-300e_640.py)| 640 | 18.3 | 64.4 | 85.4 | 78.0 | 81.5 | 11.6 | [model](),  [log]()
| [YOLOX-m](configs/body_bev_position/spiideo_scenes/yoloxpose_m_4xb64-300e_640.py)| 640 | 47.9 | 68.2 | 87.8 | 81.0 | 84.3 | 16.6 | [model](),  [log]()
| [YOLOX-tiny](configs/body_bev_position/spiideo_scenes/yoloxpose_tiny_4xb64-300e_960.py)| 960 | 23.3 | 73.3 | 88.1 | 86.0 | 87.0 | 23.1 | [model](),  [log]()
| [YOLOX-s](configs/body_bev_position/spiideo_scenes/yoloxpose_s_4xb64-300e_960.py)| 960 | 41.1 | 76.6 | 91.4 | 88.0 | 89.7 | 27.2 | [model](),  [log]()
| [YOLOX-m](configs/body_bev_position/spiideo_scenes/yoloxpose_m_4xb64-300e_960.py)| 960 | 108.0 | 79.7 | 93.1 | 89.0 | 91.0 | 30.7 | [model](),  [log]()

## Evaluation

To evaluate a pretrained model, the command below can be used. It will first run the evaluation
on the validation set and choose the final score threshold that maximizes the F1-score there. The
the model will be evaluated on the testset using choosen threshold. Results will be printen and as
well as stored in `work_dirs/<config name>/<date>_<time>/<date>_<time>.json`.

```bash
python tools/test.py configs/body_bev_position/spiideo_scenes/yoloxpose_tiny_4xb64-300e_640.py work_dirs/yoloxpose_tiny_4xb64-300e_640/epoch_300.pth
```

## Training

To train on a system with 4 GPUs, use for example

```bash
    bash ./tools/dist_train.sh configs/body_bev_position/spiideo_scenes/yoloxpose_tiny_4xb64-300e_640.py 4 --amp
```

For higher resolution inputs, the memory on the GPUs might not hold a full batch. In that case the batch can be split into multiple passes using gradient accumulation, i.e.

```bash
bash ./tools/dist_train.sh configs/body_bev_position/spiideo_scenes/yoloxpose_tiny_4xb64-300e_960.py 4 --amp --cfg-options train_dataloader.batch_size=32 optim_wrapper.accumulative_counts=2

bash ./tools/dist_train.sh configs/body_bev_position/spiideo_scenes/yoloxpose_m_4xb64-300e_960.py 4 --amp --cfg-options train_dataloader.batch_size=8 optim_wrapper.accumulative_counts=8
```

## Challenge

To evaluate on the challenge set and create a file with detections that can be submitted to the
[challenge server](), use for example:

```bash
python tools/test.py configs/body_bev_position/spiideo_scenes/yoloxpose_tiny_4xb64-300e_640.py work_dirs/yoloxpose_tiny_4xb64-300e_640/epoch_300.pth --challenge
```

This will produce `/tmp/tmp_results_locsim.keypoints.json` and `/tmp/tmp_results_locsim_val_stats.json`, which are used by the [submission script](https://github.com/Spiideo/sskit/tree/master#challenge-submission).

## Citation

If this project benefits your work, please kindly consider citing the original papers:

```bibtex
@inproceedings{maji2022yolo,
  title={YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss},
  author={Maji, Debapriya and Nagori, Soyeb and Mathew, Manu and Poddar, Deepak},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2637--2646},
  year={2022}
}
```

```bibtex
@article{yolox2021,
  title={{YOLOX}: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```

```bibtex
@misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```

Additionally, please cite our work as well:

```bibtex
...
```