import pandas as pd
from omegaconf import OmegaConf

import torch

def flatten_configdict(
    cfg: OmegaConf,
    sep: str = ".",
):
    cfgdict = OmegaConf.to_container(cfg)
    cfgdict = pd.json_normalize(cfgdict, sep=sep)

    return cfgdict.to_dict(orient="records")[0]

def entropy_param(inv_param):
    # output bs
    ub = inv_param[:,1]
    # ub = torch.fmod(color_param[:,1], 2*math.pi)
    lb = inv_param[:,0]
    # lb = torch.fmod(color_param[:,0], 2*math.pi)
    return torch.log(torch.abs(ub-lb)+1e-6)