from typing import Callable, List, Sequence
from mmengine.dataset import BaseDataset
from mmpose.registry import DATASETS
from mmpose.datasets.datasets.base import BaseCocoStyleDataset
from mmpose.datasets.datasets.body.coco_dataset import CocoDataset


@DATASETS.register_module(name='SpiideoScenes')
class SpiideoScenes(BaseCocoStyleDataset):
    METAINFO: dict = dict(from_file='configs/_base_/datasets/spiideo_scenes.py')


