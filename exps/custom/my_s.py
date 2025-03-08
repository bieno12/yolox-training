#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp
from yolox.data.datasets import COCODataset

from yolox.data.data_augment import TrainTransform, ValTransform
from mot import MOTDataset

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        
        self.num_classes=1
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "data/tracking"
        self.train_ann = "train.json"
        self.val_ann = "test.json"

        self.input_size = (608, 1088)
        self.test_size = (608, 1088)
        self.random_size = (12, 26)
        self.max_epoch = 5
        self.print_interval = 5
        self.eval_interval = 1
        self.test_conf = 0.001
        self.nmsthre = 0.7
        self.no_aug_epochs = 5
        self.basic_lr_per_img = 0.0003 / 8
        self.warmup_epochs = 1

        # aug
        # self.degrees = 10.0
        # self.translate = 0.1
        # self.scale = (0.1, 2)
        # self.mosaic_scale = (0.8, 1.6)
        # self.shear = 2.0
        # self.perspective = 0.0
        # self.enable_mixup = True
        
    def get_dataset(self, cache = False, cache_type = "ram"):
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name='test',
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=500),
            cache=cache,
            cache_type=cache_type,
        )
    
    def get_eval_dataset(self, **kwargs):
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name='test',
            img_size=self.test_size,
            preproc=ValTransform(),
        )
    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator
        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
