import torch
import torch.nn as nn

import wandb
from omegaconf import OmegaConf

from torchvision.models import resnet18
from .CEConv import models
from .color_aug import color_aug

def resnet(num_classes=2):
    network = resnet18()
    num_ftrs = network.fc.in_features
    network.fc = nn.Linear(num_ftrs, num_classes)
    return network

class InstaModel(torch.nn.Module):
    def __init__(self, model, invariance, num_samples):
        super().__init__()
        self.model = model
        self.invariance = invariance
        self.num_samples = num_samples

    def sample_param(self, inv_param, num_samples):
        bs = inv_param.shape[0]
        ub = inv_param[:,1][:,None]
        lb = inv_param[:,0][:,None]
        param = lb + torch.rand(bs,num_samples).to(inv_param.device) * (ub-lb)
        return param # [bs, num_samples]
    
    def output_with_param(self, x):
        inv_param = self.invariance(x) # [bs, 2]
        param = self.sample_param(inv_param, self.num_samples) # [bs, args.num_samples]
        x = x.unsqueeze(1).repeat(1,self.num_samples,1,1,1).view(-1,3,x.shape[2],x.shape[3]) # [bs*args.num_samples, 3, 32, 32]
        param = param.view(-1) # [bs*args.num_samples]
        transformed_images = color_aug(x, param) # [bs*args.num_samples, 3, 32, 32]
        outputs = self.model(transformed_images) # [bs*args.num_samples, num_classes]
        outputs = outputs.view(-1,self.num_samples,outputs.shape[-1]).mean(dim=1) # [bs, num_classes]
        return outputs, inv_param # [bs, num_classes], [bs, 2]
    
    def forward(self, x):
        outputs, _ = self.output_with_param(x)
        return outputs, _

def construct_model(
    cfg: OmegaConf,
) -> torch.nn.Module:
    """
    :param cfg: A config file specifying the parameters of the model.
    :return: An instance of the model specified in the config (torch.nn.Module)
    """

    if cfg.dataset in ["Flowers102"]:
        if not cfg.model.partial:
            model = models.CEResNet18(pretrained=False, progress=False, rotations=cfg.model.rot, num_classes=102,
                            groupcosetmaxpool=True, separable=True)
        else:
            model = models.CEResNet18_partial(pretrained=False, progress=False, rotations=cfg.model.rot, num_classes=102,
                            groupcosetmaxpool=True, separable=True)
        if cfg.model.insta:
            invariance = resnet(num_classes=2)
            model = InstaModel(model, invariance, num_samples=cfg.model.insta_params.num_samples)
    elif cfg.dataset in ["MNIST"]:
        if cfg.model.rot==1:
            model = models.CNN(planes=17, num_classes=30)
        elif not cfg.model.partial:
            model = models.CECNN(planes=17, rotations=cfg.model.rot, num_classes=30)
        elif cfg.model.partial:
            model = models.CECNN_partial(planes=17, rotations=cfg.model.rot, num_classes=30)
        if cfg.model.insta:
            invariance = models.CNN(planes=17, num_classes=2)
            model = InstaModel(model, invariance, num_samples=cfg.model.insta_params.num_samples)
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset} is not implemented.")
    
    # print number parameters
    no_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:", no_params)
    wandb.run.summary["no_params"] = no_params

    # Create DataParallel for multi-GPU support
    model = torch.nn.DataParallel(model)

    return model