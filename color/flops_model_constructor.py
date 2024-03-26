import torch
import torch.nn as nn
import math

import wandb
from omegaconf import OmegaConf

from torchvision.models import resnet18
from color.CEConv import models
from color.color_aug import color_aug

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
    gumbel_no_iterations = math.ceil(cfg.train_length / float(cfg.train.batch_size))  # Iter per epoch
    gumbel_no_iterations = cfg.train.epochs * gumbel_no_iterations
    invariance = None

    if cfg.dataset in ["Flowers102", "CIFAR10", "STL10"]:
        assert cfg.model.variational or ((not cfg.model.variational) and cfg.model.maxpool)
        kwargs = {"pretrained":False, "progress":False, "rotations":cfg.model.rot, "num_classes":cfg.num_classes,
                            "groupcosetmaxpool":cfg.model.maxpool, "separable":True}
        
        models_dict = {
            "resnet18": {
                "variational": models.flops_CEResNet18_variational,
                "partial": models.flops_CEResNet18_partial,
                "normal": models.flops_CEResNet18
            },
            "resnet44": {
                "variational": models.flops_CEResNet44_variational,
                "partial": models.flops_CEResNet44_partial,
                "normal": models.flops_CEResNet44
            },
        }

        if cfg.model.variational:
            model = models_dict[cfg.model.architecture]["variational"](
                **kwargs,
                gumbel_no_iterations=gumbel_no_iterations,
                version=cfg.model.version,
                vplayers=cfg.model.vplayers 
            )
        elif not cfg.model.partial:
            model = models_dict[cfg.model.architecture]["normal"](**kwargs)
        else:
            model = models_dict[cfg.model.architecture]["partial"](**kwargs)
        if cfg.model.insta:
            invariance = resnet(num_classes=2)
            model = InstaModel(model, invariance, num_samples=cfg.model.insta_params.num_samples)
    elif cfg.dataset in ["MNIST"]:
        kwargs = {"planes":17, "rotations":cfg.model.rot, "num_classes":30}
        if cfg.model.variational:
            model = models.CECNN_variational(
                **kwargs,
                gumbel_no_iterations=gumbel_no_iterations,
                version=cfg.model.version
            )
        elif cfg.model.rot==1:
            model = models.CNN(planes=17, num_classes=30)
        elif not cfg.model.partial:
            model = models.CECNN(**kwargs)
        elif cfg.model.partial:
            model = models.CECNN_partial(**kwargs)
        if cfg.model.insta:
            invariance = models.CNN(planes=17, num_classes=2)
            model = InstaModel(model, invariance, num_samples=cfg.model.insta_params.num_samples)
    # elif cfg.dataset in ["CIFAR10, STL10"]:
    #     if cfg.model.variational:
    #         model = models.CEResNet18_variational(
    #             pretrained=False, progress=False, rotations=cfg.model.rot, num_classes=10,
    #             groupcosetmaxpool=True, separable=True,
    #             gumbel_no_iterations=gumbel_no_iterations,
    #             version=cfg.model.version
    #         )
    #     elif not cfg.model.partial:
    #         model = models.CEResNet18(pretrained=False, progress=False, rotations=cfg.model.rot, num_classes=10,
    #                         groupcosetmaxpool=True, separable=True)
    #     else:
    #         model = models.CEResNet18_partial(pretrained=False, progress=False, rotations=cfg.model.rot, num_classes=10,
    #                         groupcosetmaxpool=True, separable=True)
    #     if cfg.model.insta:
    #         invariance = resnet(num_classes=2)
    #         model = InstaModel(model, invariance, num_samples=cfg.model.insta_params.num_samples)
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset} is not implemented.")
    
    # print number parameters
    no_params = sum(p.numel() for p in invariance.parameters() if p.requires_grad)
    print("Number of parameters:", no_params)
    wandb.run.summary["no_params"] = no_params
    
    if cfg.flops:
        return model

    # Create DataParallel for multi-GPU support
    model = torch.nn.DataParallel(model)

    return model