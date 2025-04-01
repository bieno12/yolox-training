# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist
from exps.custom.base_exp import BaseExp
import uuid
class Exp(BaseExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.data_dir = "data/tracking"
        self.train_ann = "train_half.json"
        self.val_ann = "val_half.json"
        self.test_ann = "test.json"
        
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.25
        
        self.mosaic_scale = (0.5, 1.5)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.input_size = (608, 1088)
        self.test_size = (608, 1088)
        self.random_size = (12, 26)
        self.max_epoch = 80
        self.print_interval = 20
        self.eval_interval = 5
        self.test_conf = 0.001
        self.nmsthre = 0.7
        self.no_aug_epochs = 10
        self.warmup_epochs = 1
        
        self.basic_lr_per_img = 0.001 / 64
        self.output_dir = "./MOT20"
        self.seed = uuid.uuid4().int % 2**32
        self.set_envvars()
    
    def set_envvars(self):
        for key, value in os.environ.items():
            if key.startswith("EXP_PROP_"):
                prop_name = key[9:].lower()
                if hasattr(self, prop_name):
                    current_value = getattr(self, prop_name)
                    try:
                        if isinstance(current_value, bool):
                            new_value = value.lower() == "true"
                        elif isinstance(current_value, (list, tuple)):
                            new_value = type(current_value)(eval(value))
                        else:
                            new_value = type(current_value)(value)
                        setattr(self, prop_name, new_value)
                    except (ValueError, SyntaxError) as e:
                        print(f"Warning: Could not convert {key}={value} to type {type(current_value).__name__}: {str(e)}")
    
    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from swyolox.models import YOLOX, SWYOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = SWYOLOPAFPN(self.depth, self.width, in_channels=in_channels, depthwise=True, use_PE=True)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, depthwise=True)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
