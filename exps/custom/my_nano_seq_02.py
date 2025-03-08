# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as MyExp
# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist



class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.data_dir = "data/tracking"
        self.train_ann = "train_seq_02.json"
        self.val_ann = "test.json"
        
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.25
        self.scale = (0.5, 1.5)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.input_size = (608, 1088)
        self.test_size = (608, 1088)
        #overfit configuration
        self.ema = True
        self.max_epoch = 30
        self.warmup_epochs = 5  # Gradually increase learning rate to avoid instability
        self.no_aug_epochs = 15 


        self.warmup_lr = 1e-6  # Very low LR at the start to prevent sudden jumps
        self.min_lr_ratio = 0.05  # Allows gradual learning rate decay
        self.basic_lr_per_img = 0.001 / 8  # Standard learning rate per image (adjustable with batch size)

        self.random_size = (12, 26)
        self.test_conf = 0.01
        self.nmsthre = 0.7
        
        self.print_interval = 10  # Log every iteration for detailed monitoring
        self.eval_interval = 2 # Evaluate every epoch

        self.save_history_ckpt = False
        
    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, depthwise=True)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, depthwise=True)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
    def get_dataset(self, cache = False, cache_type = "ram"):
        from yolox.data import COCODataset, TrainTransform
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name='train',
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=500),
            cache=cache,
            cache_type=cache_type,
        )
    
    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name='test',
            img_size=self.test_size,
            preproc=ValTransform(legacy=kwargs.get('legacy', False)),
        )
    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator
        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev, legacy=legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator