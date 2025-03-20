# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as BaseExp

class Exp(BaseExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.data_dir = "data/tracking"
        self.train_ann = "train_half.json"
        self.val_ann = "val_half.json"
        self.test_ann = "test.json"
        self.batch_size = 8
        
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.input_size = (608, 1088)
        self.test_size = (608, 1088)
        #overfit configuration
        self.ema = True
        self.max_epoch = 30
        self.warmup_epochs = 5  # Gradually increase learning rate to avoid instability
        self.no_aug_epochs = 5 


        self.warmup_lr = 1e-6  # Very low LR at the start to prevent sudden jumps
        self.min_lr_ratio = 0.05  # Allows gradual learning rate decay
        self.basic_lr_per_img = 0.001 / self.batch_size # Standard learning rate per image (adjustable with batch size)

        self.random_size = (6, 12)
        self.test_conf = 0.1
        self.nmsthre = 0.7
        
        self.mosaic_prob = 1.0
        self.enable_mixup = True
        self.print_interval = 40  # Log every iteration for detailed monitoring
        self.eval_interval = 2 # Evaluate every epoch

        self.save_history_ckpt = False
        self.max_labels = 100
        #override using env vars
        self.set_envvars()
    
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
        transform = TrainTransform(max_labels=self.max_labels)
            
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name='train',
            img_size=self.input_size,
            preproc=transform,
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.test_ann if kwargs.get('testdev', False) else self.val_ann,
            name='test' if kwargs.get('testdev', False) else 'train',
            img_size=self.test_size,
            preproc=ValTransform(legacy=self.legacy),
        )
