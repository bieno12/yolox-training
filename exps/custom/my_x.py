# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        
        self.data_dir = "data/tracking"
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.batch_size = 4
        self.test_ann = "test.json"
        
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        self.random_size = (18, 32)
        
        self.max_epoch = 30
        self.no_aug_epochs = 15
        self.warmup_epochs = 5
        self.basic_lr_per_img = 0.001 / self.batch_size # Standard learning rate per image (adjustable with batch size)
        
        self.print_interval = 20
        self.eval_interval = 2
        
        self.test_conf = 0.01
        self.nmsthre = 0.7
        
        self.save_history_ckpt = False
        
        self.legacy = False
        #override using env vars
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
                        
    def get_dataset(self, cache = False, cache_type = "ram"):
        from yolox.data import COCODataset, TrainTransform

        if self.legacy:
            from legacy import TrainTransform
            transform = TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            )
        else:
            transform = TrainTransform(max_labels=500)
            
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

