"""Generic detector."""
import os
import pickle

import torch

import warnings

import torch
import torch.nn as nn

from yolox.exp import get_exp
from yolox.utils import postprocess, fuse_model


class PostModel(nn.Module):
    def __init__(self, model, exp):
        super().__init__()
        self.exp = exp
        self.model = model

    def forward(self, batch):
        """
        Returns Nx5, (x1, y1, x2, y2, conf)
        """
        raw = self.model(batch)
        pred = postprocess(
            raw, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre
        )[0]
        if pred is not None:
            return torch.cat((pred[:, :4], (pred[:, 4] * pred[:, 5])[:, None]), dim=1)
        else:
            return None


def get_model_with_exp(exp_path:str, ckpt_path:str):
    exp = get_exp(exp_path)
    model = exp.get_model()
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model"])
    with warnings.catch_warnings():
        model = fuse_model(model)
    model = model.half()
    model = PostModel(model, exp)
    model.cuda()
    model.eval()
    
    return model

class YOLOXDetector(torch.nn.Module):

    def __init__(self, exp_path:str, ckpt_path:str):
        super().__init__()

        self.exp_path = exp_path
        self.model_path = ckpt_path
        self.model = None
    
        os.makedirs("./cache", exist_ok=True)
        self.cache_path = os.path.join(
            "./cache", f"det_{os.path.basename(ckpt_path).split('.')[0]}.pkl"
        )
        self.cache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as fp:
                self.cache = pickle.load(fp)
        self.initialize_model()

    def initialize_model(self):
        self.model = get_model_with_exp(self.exp_path, self.model_path)

    def forward(self, batch, tag=None):
        if tag in self.cache:
            return self.cache[tag]
        if self.model is None:
            self.initialize_model()

        with torch.no_grad():
            batch = batch.half()
            output = self.model(batch)
        if output is not None:
            self.cache[tag] = output.cpu().detach()

        return output

    def dump_cache(self):
        with open(self.cache_path, "wb") as fp:
            pickle.dump(self.cache, fp)
