from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval
import numpy as np
from sskit.coco import LocSimCOCOeval, BBoxLocSimCOCOeval
import time

class LocSimCOCOeval(LocSimCOCOeval):
    def accumulate(self, p=None):
        if p is None:
            p = self.params
        super().accumulate(p)

        iou = p.iouThrs == 0.5
        area = p.areaRngLbl.index('all')
        dets = np.argmax(p.maxDets)

        precision = np.squeeze(self.eval['precision'][iou, :, 0, area, dets])
        scores = np.squeeze(self.eval['scores'][iou, :, 0, area, dets])
        recall = p.recThrs
        f1 = 2 * precision * recall / (precision + recall)

        self.eval['precision_50'] = precision
        self.eval['recall_50'] = recall
        self.eval['f1_50'] = f1
        self.eval['scores_50'] = scores

    def summarize(self):
        super().summarize()
        i = self.eval['f1_50'].argmax()
        stats = [self.eval['precision_50'][i], self.eval['recall_50'][i], self.eval['f1_50'][i], self.eval['scores_50'][i]]
        self.stats = np.concatenate([self.stats, stats])



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

    coco_eval.eval['precision_50'].shape
    coco_eval.eval['f1_50'].max()
    len(coco_eval.stats)

    coco_eval.eval['scores'].shape
    coco_eval.eval['precision'].shape
    coco_eval.eval['recall'].shape
    len(coco_eval.params.iouThrs) # T
    len(coco_eval.params.recThrs) # R
    len(coco_eval.params.catIds)  # K
    len(coco_eval.params.areaRng) # A
    len(coco_eval.params.maxDets) # M

    iou = coco_eval.params.iouThrs == 0.5
    coco_eval.params.catIds
    area = coco_eval.params.areaRngLbl.index('all')
    dets = np.argmax(coco_eval.params.maxDets)


    precision = np.squeeze(coco_eval.eval['precision'][iou, :, 0, area, dets])
    scores = np.squeeze(coco_eval.eval['scores'][iou, :, 0, area, dets])
    recall = coco_eval.params.recThrs
    f1 = 2 * precision * recall / (precision + recall)
    f1.max()
    threshold = scores[f1.argmax()]

    i = np.searchsorted(-scores, -threshold, 'right') - 1
    scores[i]
    f1[i]
    precision[i]
    recall[i]

    len(coco_eval.stats)