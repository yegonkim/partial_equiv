import time
import torch
from color.ada_aug import utils
import torch.nn as nn
import torch.utils
import wandb
from omegaconf import OmegaConf

from color.ada_aug.adaptive_augmentor import AdaAug
from color.ada_aug.networks import get_model
from color.ada_aug.networks.projection import Projection
from color.ada_aug.dataset import get_dataloaders


def train_AdaAug(dataset, cfg: OmegaConf) -> AdaAug:
    #  dataset settings
    n_class = cfg.num_classes
    # sdiv = get_search_divider(cfg.ada_aug.model_name)
    # class2label = get_label_name(cfg.dataset)

    train_queue, search_queue = get_dataloaders(dataset, cfg)

    #  model settings
    gf_model = get_model(model_name=cfg.ada_aug.model_name, num_class=n_class, cfg=cfg)
    # logging.info("param size = %fMB", utils.count_parameters_in_MB(gf_model))

    h_model = Projection(in_features=gf_model.fc.in_features, n_layers=0, n_hidden=128).to(cfg.device)

    #  training settings
    gf_optimizer = torch.optim.Adam(
        gf_model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay)
    
    gf_optimizer = torch.optim.SGD(
        gf_model.parameters(),
        0.1,
        momentum=0.9,
        weight_decay=0.0001)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(gf_optimizer, T_max=cfg.ada_aug.epochs, eta_min=0.001)

    h_optimizer = torch.optim.Adam(
        h_model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(cfg.device)

    #  AdaAug settings
    dataset_dimension = dataset[0][0].shape[1]
    # print("dataset_dimension", dataset_dimension)
    adaaug_config = {'sampling': 'prob',
                    'k_ops': 1,
                    'delta': 0.0,
                    'temp': 1.0,
                    'search_d': dataset_dimension,
                    'target_d': dataset_dimension}

    adaaug = AdaAug(
        n_class=n_class,
        gf_model=gf_model,
        h_model=h_model,
        save_dir=wandb.run.dir,
        config=adaaug_config,
        cfg=cfg
        )

    #  Start training
    # start_time = time.time()
    for epoch in range(cfg.ada_aug.epochs):
        lr = scheduler.get_last_lr()[0]
        # logging.info('epoch %d lr %e', epoch, lr)

        # searching
        train_acc, train_obj = train(train_queue, search_queue, gf_model, adaaug,
            criterion, gf_optimizer, cfg.ada_aug.grad_clip, h_optimizer, epoch, cfg.ada_aug.search_freq, cfg)
        
        # logging.info(f'train_acc {train_acc}')
        scheduler.step()
        wandb.log({"ada_aug/train_acc": train_acc, "ada_aug/train_obj": train_obj, "lr": lr})
        # print(f'train_acc {train_acc}')

        # utils.save_model(gf_model, os.path.join(args.save, 'gf_weights.pt'))
        # utils.save_model(h_model, os.path.join(args.save, 'h_weights.pt'))

    return adaaug


def train(train_queue, search_queue, gf_model, adaaug, criterion, gf_optimizer,
            grad_clip, h_optimizer, epoch, search_freq, cfg):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        input = input.to(cfg.device)
        target = target.to(cfg.device)

        # exploitation
        timer = time.time()
        aug_images = adaaug(input, mode='exploit')
        gf_model.train()
        gf_optimizer.zero_grad()
        logits = gf_model(aug_images)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(gf_model.parameters(), grad_clip)
        gf_optimizer.step()

        #  stats
        prec1, prec5 = utils.accuracy(logits.detach(), target.detach(), topk=(1, 5))
        n = target.size(0)
        objs.update(loss.detach().item(), n)
        top1.update(prec1.detach().item(), n)
        top5.update(prec5.detach().item(), n)
        exploitation_time = time.time() - timer

        # exploration
        timer = time.time()
        if step % search_freq == 0:
            input_search, target_search = next(iter(search_queue))
            input_search = input_search.to(cfg.device)
            target_search = target_search.to(cfg.device)

            h_optimizer.zero_grad()
            mixed_features = adaaug(input_search, mode='explore')
            logits = gf_model.g(mixed_features)
            loss = criterion(logits, target_search)
            loss.backward()
            h_optimizer.step()
            exploration_time = time.time() - timer

            #  log policy
            adaaug.add_history(input_search, target_search)

        global_step = epoch * len(train_queue) + step
        # if global_step % args.report_freq == 0:
        #     logging.info('  |train %03d %e %f %f | %.3f + %.3f s', global_step,
        #         objs.avg, top1.avg, top5.avg, exploitation_time, exploration_time)

    return top1.avg, objs.avg


# def infer(valid_queue, gf_model, criterion):
#     objs = utils.AvgrageMeter()
#     top1 = utils.AvgrageMeter()
#     top5 = utils.AvgrageMeter()
#     gf_model.eval()

#     with torch.no_grad():
#         for input, target in valid_queue:
#             input = input.to(cfg.device)
#             target = target.cuda(non_blocking=True)

#             logits = gf_model(input)
#             loss = criterion(logits, target)

#             prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
#             n = input.size(0)
#             objs.update(loss.detach().item(), n)
#             top1.update(prec1.detach().item(), n)
#             top5.update(prec5.detach().item(), n)

#     return top1.avg, objs.avg

