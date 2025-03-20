# encoding: utf-8
import os
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as MyExp

class BaseExp(MyExp):
    def __init__(self):
        super(BaseExp, self).__init__()
        
        self.seed = 42
        
        self.data_dir = "data/tracking"
        self.train_ann = "train_half.json"
        self.val_ann = "val_half.json"
        self.batch_size = 4
        self.test_ann = "test.json"
        self.save_history_ckpt = False
        self.num_classes = 1
        self.test_conf = 0.1
        self.nmsthre = 0.7
        self.legacy = False
        self.max_labels = 230
        self.mosaic_scale = (0.8, 1.3)
        self.mixup_scale = (0.8, 1.2)

    
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
                max_labels=self.max_labels,
            )
        else:
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

