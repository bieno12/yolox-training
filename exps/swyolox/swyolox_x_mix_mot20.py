# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist
from exps.custom.base_exp import BaseExp

class Exp(BaseExp):
    def __init__(self):
        super(Exp, self).__init__()
        
        self.seed = 42
        
        self.data_dir = "data/mix_mot20_ch"
        self.train_ann = "train.json"
        self.val_ann = "val_half.json"
        self.batch_size = 4
        self.test_ann = "test.json"
        
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.input_size = (896, 1600)
        self.test_size = (896, 1600)
        
        self.random_size = (28, 36)
        
        self.max_epoch = 30
        self.no_aug_epochs = 10
        self.warmup_epochs = 1
        self.basic_lr_per_img = 0.001 / 64 
        self.mosaic_scale = (0.8, 1.2)
        self.mixup_scale = (0.8, 1.6)
        
        self.print_interval = 100
        
        self.eval_interval = 3
        
        self.test_conf = 0.001
        self.nmsthre = 0.7
        
        self.save_history_ckpt = False
        
        # my props
        self.legacy = False
        self.max_labels = 100
        

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
    
    def get_model(self):
        from swyolox.models import YOLOX, SWYOLOPAFPN, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = SWYOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model
                        
    def get_dataset(self, cache = False, cache_type = "ram"):
        from yolox.data import COCODataset, TrainTransform

        if self.legacy:
            from legacy import TrainTransform
            transform = TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=self.max_labels,
            )
        else:
            transform = TrainTransform(max_labels=self.max_labels)
            
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name='',
            img_size=self.input_size,
            preproc=transform,
            cache=cache,
            cache_type=cache_type,
        )

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img: str = None):
        """
        Get dataloader according to cache_img parameter.
        Args:
            no_aug (bool, optional): Whether to turn off mosaic data enhancement. Defaults to False.
            cache_img (str, optional): cache_img is equivalent to cache_type. Defaults to None.
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
                None: Do not use cache, in this case cache_data is also None.
        """
        from yolox.data import (
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        # if cache is True, we will create self.dataset before launch
        # else we will create self.dataset after launch
        if self.dataset is None:
            with wait_for_the_master():
                assert cache_img is None, \
                    "cache_img must be None if you didn't create self.dataset before launch"
                self.dataset = self.get_dataset(cache=False, cache_type=cache_img)

        self.dataset = MosaicDetection(
            dataset=self.dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=self.max_labels*4,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader
    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.test_ann if kwargs.get('testdev', False) else self.val_ann,
            name='test' if kwargs.get('testdev', False) else 'train',
            img_size=self.test_size,
            preproc=ValTransform(legacy=self.legacy),
        )

