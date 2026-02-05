import os
import numpy as np
import sys

import torch
sys.path.append('src/')

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def GPU_Search():
    os.system("nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >curtmp")
    memory_gpu = [int(x.split()[2]) for x in open('curtmp', 'r').readlines()]
    os.system("rm curtmp")
    return np.argmax(memory_gpu)


def torch_save(model, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu(), save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path, weights_only=False)
    if device is not None:
        model = model.to(device)
    return model


def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, 'to'):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)