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
        threshold = (self.eval['scores_50'][i] + self.eval['scores_50'][i+1]) / 2
        stats = [self.eval['precision_50'][i], self.eval['recall_50'][i], self.eval['f1_50'][i], threshold]
        if hasattr(self.params, 'score_threshold'):
            threshold = self.params.score_threshold
        else:
            threshold = 0.5
        i = np.searchsorted(-self.eval['scores_50'], -threshold, 'right') - 1
        stats += [self.eval['precision_50'][i], self.eval['recall_50'][i], self.eval['f1_50'][i], threshold]
        self.stats = np.concatenate([self.stats, stats])



if __name__ == '__main__':
    coco = COCO('data/SpiideoScenes/Soccer/v1/annotations/test.json')
    coco_det = coco.loadRes("tstres.keypoints_run1.json")

    # coco_eval = COCOeval(coco, coco_det, 'bbox', [0.089, 0.089], True)
    coco_eval = LocSimCOCOeval(coco, coco_det, 'bbox', [0.089, 0.089], True)
    # coco_eval = BBoxLocSimCOCOeval(coco, coco_det, 'bbox', [0.089, 0.089], True)


    coco_eval.params.useSegm = None
    coco_eval.params.imgIds = [0]
    coco_eval.params.score_threshold = 0.1


    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print(coco_eval.stats[-8:-4])
    print(coco_eval.stats[-4:])
