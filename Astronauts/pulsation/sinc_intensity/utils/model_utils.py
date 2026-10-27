import torch
import os
import re
import sys


def natsorted(names):
    # natsort n'est pas installe dans pyOR : tri naturel suffisant pour '<model>_e<epoch>'.
    return sorted(names, key=lambda n: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', n)])


def get_last_checkpoint(save_root):
    model_files = natsorted(os.listdir(save_root))
    last_epoch = len(model_files) - 1
    checkpoint = os.path.join(save_root, model_files[-1])
    return checkpoint, last_epoch


def get_best_loss(best_save_root):
    model_files = os.listdir(best_save_root)
    if len(model_files) != 1:
        print('Zero or more than one best model when trying to load best model. Exiting.')
        sys.exit(-1)
    checkpoint_path = os.path.join(best_save_root, model_files[0])
    checkpoint = torch.load(checkpoint_path)
    best_loss = checkpoint['loss']['total']
    return best_loss


