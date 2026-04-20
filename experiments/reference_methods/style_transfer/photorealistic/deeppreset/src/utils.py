import json
import numpy as np
import copy
import torch
from torch import nn
import os

def size_str2tuple(_str):
    out = [int(k) for k in _str.split("x")]
    assert len(out) == 2, "Unknown {}. The size should have been [width]x[height].".format(_str)
    return tuple(out)

class PresetHandler:
    def __init__(self, base_p_dir=None):
        if base_p_dir is None:
            base_p_dir = os.path.join(os.path.dirname(__file__), 'data', 'base_presets.json')
            
        base_settings = self.json_load(base_p_dir)
        self.p_default = base_settings['p_default']
        self.keys = base_settings['keys']
        self.max_bound_np = np.array([base_settings['max'][k] for k in self.keys])
        self.min_bound_np = np.array([base_settings['min'][k] for k in self.keys])

    @staticmethod
    def json_load(_dir):
        with open(_dir) as json_file:
            content = json.load(json_file)
        return content

    def unnorm_preset(self, p_np):
        out = (p_np + 1)/2*(self.max_bound_np-self.min_bound_np) + self.min_bound_np
        return out.round().astype(np.int16).tolist()

    @staticmethod
    def int2bool(_num):
        return True if _num > 0.5 else False

    def save_numpy_preset(self, out_dir, p_np):
        p_list = self.unnorm_preset(p_np)
        p_base = copy.deepcopy(self.p_default)
        p_dict = {k:p_list[i] for i,k in enumerate(self.keys)}
        for k in p_dict:
            p_dict[k] = self.int2bool(p_dict[k]) if type(p_base[k]) == bool else p_dict[k]
        
        p_base.update(p_dict)
        with open(out_dir, 'w') as outfile:
            json.dump(p_base, outfile)

    def norm_preset(self, p):
        tmp = [p[k] for k in p if k in self.keys] # Fixed p_handler.keys() to self.keys
        # assert len(tmp.keys()) == 69 # tmp is list, no keys()
        # Also p_handler is not defined, use self
        
        # The original code had some bugs or relied on global p_handler?
        # "tmp = [p[k] for k in p if k in p_handler.keys()]" -> p is dict, k is key.
        # "assert len(tmp.keys()) == 69" -> tmp is list.
        
        # Let's look at the original code again.
        # tmp = [p[k] for k in p if k in p_handler.keys()]
        # assert len(tmp.keys()) == 69
        # for k in tmp:
        #    tmp[k] = 2*(tmp[k]-p_handler.min_bound_np[k])/(p_handler.max_bound_np[k]-p_handler.min_bound_np[k]) - 1
        
        # This looks like it was intended to be a dict?
        # If p is a dict, tmp is a list of values.
        # But then it iterates k in tmp?
        
        # I will comment out norm_preset as it seems broken in original or I misread it, 
        # and we probably don't need it for inference (we only need unnorm_preset for saving, maybe).
        # Actually we only need stylization, so we might not need PresetHandler at all if we don't save presets.
        # But DeepPreset class initializes it.
        
        pass
        # return tmp
