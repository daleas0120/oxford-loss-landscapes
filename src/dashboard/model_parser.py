import glob
import os
import pandas as pd
import numpy as np
import torch

from src.experiment import load_model

def get_list_of_models(MODEL_DIR: str):
    data_path = os.path.join(MODEL_DIR, '**1.00_0.00_*/checkpoint_25.pt')

    return glob.glob(data_path)


def parse_model_layers(model_path):
    cfg = '/home/ashley/lollaa/configs/kangming_config.json'
    model = load_model(cfg, model_path, verbose=False)
    model_dict = {}

    list_of_layers = []
    for name, _ in model.named_parameters():
        if name.split('.')[-1] == 'weight':
            list_of_layers.append({'label': name, 'value':name})
    return list_of_layers


def get_model_layer_params(model_path, layer_name):
    cfg = '/home/ashley/lollaa/configs/kangming_config.json'
    model = load_model(cfg, model_path, 'cpu', verbose=False)
    params = model.state_dict()[layer_name].detach().numpy()
    if len(params.shape) < 2:
        params = np.expand_dims(params, axis=0)
    if len(params.shape) > 2:
        params = np.reshape(params, (params.shape[0], -1))
    return params
    # for name, param in model.named_parameters():
    #     if name.split('.')[-1] == 'weight':
    #     # print(name)
    #         model_dict[name] = param.detach().numpy()

    # return model_dict
