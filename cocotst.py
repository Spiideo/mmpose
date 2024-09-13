from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval
import numpy as np
from sskit.coco import LocSimCOCOeval, BBoxLocSimCOCOeval

if __name__ == '__main__':
    coco = COCO('data/SpiideoScenes/Soccer/v1/annotations/test.json')
    coco_det = coco.loadRes("tstres.keypoints_run1.json")

    # coco_eval = COCOeval(coco, coco_det, 'bbox', [0.089, 0.089], True)
    coco_eval = LocSimCOCOeval(coco, coco_det, 'bbox', [0.089, 0.089], True)
    # coco_eval = BBoxLocSimCOCOeval(coco, coco_det, 'bbox', [0.089, 0.089], True)


    coco_eval.params.useSegm = None
    coco_eval.params.imgIds = [0]


    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
