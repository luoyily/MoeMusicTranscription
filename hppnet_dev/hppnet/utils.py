from pickletools import uint8
import sys
from functools import reduce

import torch
from PIL import Image
from torch.nn.modules.module import _addindent

import os
from posixpath import basename
import numpy as np
import inspect
from shutil import copyfile


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def summary(model, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, 'shape'):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stdout:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        if isinstance(file, str):
            file = open(file, 'w')
        print(string, file=file)
        file.flush()

    return count


def save_pianoroll(path, onsets, frames, onset_threshold=0.5, frame_threshold=0.5, zoom=4):
    """
    Saves a piano roll diagram

    Parameters
    ----------
    path: str
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float
    zoom: int
    """
    onsets = (1 - (onsets.t() > onset_threshold).to(torch.uint8)).cpu()
    frames = (1 - (frames.t() > frame_threshold).to(torch.uint8)).cpu()
    both = (1 - (1 - onsets) * (1 - frames))
    image = torch.stack([onsets, frames, both], dim=2).flip(0).mul(255).numpy()
    image = Image.fromarray(image, 'RGB')
    image = image.resize((image.size[0], image.size[1] * zoom))
    image.save(path)


def save_pianoroll_overlap(path, frames_label, frames_pred, frame_threshold=0.5, zoom=4):
    """
    Saves a piano roll diagram
    overlap lable and pred frames together.

    Parameters
    ----------
    path: str
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float
    zoom: int
    """
    frames_label = (1 - frames_label.t().to(torch.uint8)).cpu()
    frames_pred = (1-(frames_pred.t() > frame_threshold).to(torch.uint8)).cpu()
    both = frames_label - frames_label + 1

    mask = (frames_label == 0).int() * (frames_pred == 0).int()
    mask = mask.bool().flip(0) # reverse frequency dimention. 
    # mask = torch.stack([mask, mask, mask], dim=2)
    # => [H x W x 3]
    image = torch.stack([frames_label, frames_pred, both], dim=2).flip(0).mul(255).numpy()
    # change color of True Positive from blue[0,0,1] to gray[0.8,0.8,0.8] 
    
    image[mask] = int(255*0.8) 
    image = Image.fromarray(image, 'RGB')
    image = image.resize((image.size[0], image.size[1] * zoom))
    image.save(path)


def save_src_files(elems_dict, dest_dir, query_str = 'pytorch/', src_path_set = set()):
    for key, x in elems_dict.items():
        if(inspect.isfunction(x) or inspect.ismodule(x) or inspect.isclass(x)):
            try:
                src_path = inspect.getfile(x)
                rel_path = os.path.relpath(src_path)

                if(rel_path.find(query_str) >= 0 or rel_path.find('../') < 0):
                    if(rel_path in src_path_set):
                        continue
                    src_path_set.add(rel_path)

                    dest_path = os.path.join(dest_dir, rel_path)
                    abs_dir = os.path.split(dest_path)[0]
                    os.makedirs(abs_dir, exist_ok=True)
                    copyfile(src_path, dest_path)
                    
                    
                    print(rel_path)
                    y = __import__(x.__module__)
                    new_elem_dict = vars(y)
                    save_src_files(new_elem_dict, dest_dir, query_str, src_path_set)
            except:
                pass

def copy_dir(src_dir, dest_dir, file_type = '.py'):
    for item in os.scandir(src_dir):
        if item.is_dir():
            pass
        elif item.is_file():
            name, ext = os.path.splitext(item.path)
            if(ext == file_type):
                relative_path = os.path.relpath(item.path, src_dir)
                dest_path = os.path.join(dest_dir, relative_path)
                os.makedirs(os.path.split(dest_path)[0], exist_ok=True)
                copyfile(item.path, dest_path)