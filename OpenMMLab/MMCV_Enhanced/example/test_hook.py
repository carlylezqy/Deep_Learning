from mmcv.utils import Registry
from mmcv.runner.hooks import Hook
from torchmetrics.classification import MulticlassAccuracy

import sys
sys.path.append('/home/akiyo/nfs/zhang/github/Deep_Learning/OpenMMLab/MMClassification_Enhanced')

from vis_result import show_images_and_labels

HOOKS = Registry('hook')
@HOOKS.register_module()
class TestHook(Hook):
    def __init__(self):
        pass
    
    def before_train_epoch(self, runner):
        self.before_epoch(runner)
        self.CLASSES = runner.data_loader.dataset.CLASSES
        self.metric = MulticlassAccuracy(num_classes=len(self.CLASSES))

    def after_iter(self, runner):
        if self.every_n_iters(runner, 100):
            
            img      = runner.data_batch['img']
            gt_label = runner.data_batch['gt_label']
            predicts = runner.outputs['predicts'].argmax(dim=1)
            
            pred_label = list(map(lambda x, y: f"{self.CLASSES[x]}/{self.CLASSES[y]}", predicts, gt_label))

            path = runner.work_dir + f"/pred_{runner.iter}.png"
            show_images_and_labels(img, pred_label, output_path=path)

            value = self.metric(predicts.cpu(), gt_label.cpu()).item()
            runner.log_buffer.output['Accurary'] = value
            runner.log_buffer.output['loss'] = runner.outputs['loss'].item()