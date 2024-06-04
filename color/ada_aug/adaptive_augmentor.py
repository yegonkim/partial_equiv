import random
# from cv2 import magnitude
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


from color.ada_aug.operation import apply_augment
from color.ada_aug.networks import get_model
from color.ada_aug.utils import PolicyHistory
from color.ada_aug.config import OPS_NAMES

default_config = {'sampling': 'prob',
                    'k_ops': 1,
                    'delta': 0,
                    'temp': 1.0,
                    'search_d': 32,
                    'target_d': 32}

def perturb_param(param, delta):
    if delta <= 0:
        return param
        
    amt = random.uniform(0, delta)
    if random.random() < 0.5:
        return max(0, param-amt)
    else:
        return min(1, param+amt)

def stop_gradient(trans_image, magnitude):
    images = trans_image
    adds = 0

    images = images - magnitude
    adds = adds + magnitude
    images = images.detach() + adds
    return images

class AdaAug(nn.Module):
    def __init__(self, n_class, gf_model, h_model, save_dir=None, 
                    config=default_config, cfg=None):
        super(AdaAug, self).__init__()
        self.ops_names = OPS_NAMES
        self.n_ops = len(self.ops_names)
        self.save_dir = save_dir
        self.gf_model = gf_model
        self.h_model = h_model
        self.n_class = n_class
        self.resize = config['search_d'] != config['target_d']
        self.search_d = config['search_d']
        self.k_ops = config['k_ops']
        self.sampling = config['sampling']
        self.temp = config['temp']
        self.delta = config['delta']
        self.history = PolicyHistory(self.ops_names, self.save_dir, self.n_class)
        self.cfg = cfg
        self.device = cfg.device

    def save_history(self, class2label=None):
        self.history.save(class2label)

    def plot_history(self):
        return self.history.plot()
    
    def predict_aug_params(self, images, mode):
        self.gf_model.eval()
        if mode == 'exploit':
            self.h_model.eval()
            T = self.temp
        elif mode == 'explore':
            self.h_model.train()
            T = 1.0
        # images = images.to(self.cfg.device)
        a_params = self.h_model(self.gf_model.f(images))
        magnitudes, weights = torch.split(a_params, self.n_ops, dim=1)
        magnitudes = torch.sigmoid(magnitudes)
        weights = torch.nn.functional.softmax(weights/T, dim=-1)
        return magnitudes, weights

    def add_history(self, images, targets):
        magnitudes, weights = self.predict_aug_params(images, 'exploit')
        for k in range(self.n_class):
            idxs = (targets == k).nonzero().squeeze()
            mean_lambda = magnitudes[idxs].mean(0).detach().cpu().tolist()
            mean_p = weights[idxs].mean(0).detach().cpu().tolist()
            std_lambda = magnitudes[idxs].std(0).detach().cpu().tolist()
            std_p = weights[idxs].std(0).detach().cpu().tolist()
            self.history.add(k, mean_lambda, mean_p, std_lambda, std_p)

    def get_aug_valid_imgs(self, images, magnitudes):
        """Return the mixed latent feature

        Args:
            images ([Tensor]): [description]
            magnitudes ([Tensor]): [description]
        Returns:
            [Tensor]: a set of augmented validation images
        """
        trans_image_list = []
        for i, image in enumerate(images):
            # pil_img = transforms.ToPILImage()(image)
            # Prepare transformed image for mixing
            for k, ops_name in enumerate(self.ops_names):
                trans_image = apply_augment(image, ops_name, magnitudes[i][k])
                # trans_image = stop_gradient(trans_image.to(self.cfg.device), magnitudes[i][k])
                trans_image = stop_gradient(trans_image, magnitudes[i][k])
                trans_image_list.append(trans_image)
        return torch.stack(trans_image_list, dim=0)

    def explore(self, images):
        """Return the mixed latent feature

        Args:
            images ([Tensor]): [description]
        Returns:
            [Tensor]: return a batch of mixed features
        """
        magnitudes, weights = self.predict_aug_params(images, 'explore')
        a_imgs = self.get_aug_valid_imgs(images, magnitudes)
        a_features = self.gf_model.f(a_imgs)
        ba_features = a_features.reshape(len(images), self.n_ops, -1)
        
        mixed_features = [w.matmul(feat) for w, feat in zip(weights, ba_features)]
        mixed_features = torch.stack(mixed_features, dim=0)
        return mixed_features

    def get_training_aug_images(self, images, magnitudes, weights):
        # visualization
        if self.k_ops > 0:
            trans_images = []
            if self.sampling == 'prob':
                idx_matrix = torch.multinomial(weights, self.k_ops)
            elif self.sampling == 'max':
                idx_matrix = torch.topk(weights, self.k_ops, dim=1)[1]

            for i, image in enumerate(images):
                # pil_image = transforms.ToPILImage()(image)
                for idx in idx_matrix[i]:
                    m_pi = perturb_param(magnitudes[i][idx], self.delta)
                    trans_image = apply_augment(image, self.ops_names[idx], m_pi)
                trans_images.append(image)
        else:
            trans_images = []
            for i, image in enumerate(images):
                # trans_image = transforms.ToPILImage()(image)
                trans_images.append(image)
        
        # aug_imgs = torch.stack(trans_images, dim=0).to(self.cfg.device)
        aug_imgs = torch.stack(trans_images, dim=0)
        return aug_imgs

    def exploit(self, images):
        resize_imgs = F.interpolate(images, size=self.search_d) if self.resize else images
        magnitudes, weights = self.predict_aug_params(resize_imgs, 'exploit')
        # print("magnitudes", magnitudes[0])
        # print("weights", weights[0])
        aug_imgs = self.get_training_aug_images(images, magnitudes, weights)
        return aug_imgs

    def forward(self, images, mode):
        if mode == 'explore':
            #  return a set of mixed augmented features
            return self.explore(images)
        elif mode == 'exploit':
            #  return a set of augmented images
            return self.exploit(images)
        elif mode == 'inference':
            return images
