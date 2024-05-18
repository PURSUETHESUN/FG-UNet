import math

import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target,dice_coeff
#from .losses import train_loss
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score, confusion_matrix

def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    # losses = {}
    # for name, x in inputs.items():
    loss = nn.functional.cross_entropy(inputs, target, ignore_index=ignore_index, weight=loss_weight)
    if dice is True:
        dice_target = build_target(target, num_classes, ignore_index)
        loss += dice_loss(inputs, dice_target, multiclass=True, ignore_index=ignore_index)
    return loss

    # losses[name] = loss
    # if len(losses) == 1:
    #     return losses['out']
    #
    # return losses['out'] + 0.5 * losses['aux']

def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None

    # #the weight of bce and dice
    # loss_weight = [1.,1.]

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            lateral_map_1 = model(image)
            # loss3 = criterion(lateral_map_3, target, loss_weight=loss_weight, ignore_index = 255)
            # loss2 = criterion(lateral_map_2, target,  loss_weight=loss_weight, ignore_index = 255)
            loss1 = criterion(lateral_map_1, target, loss_weight=loss_weight, ignore_index=255)
            # losse_edg = dice_coeff(torch.sigmoid(edge), torch.unsqueeze(target, dim=1).float(), ignore_index=255.)
            # losse_obj = dice_coeff(torch.sigmoid(obj), torch.unsqueeze(target, dim=1).float(), ignore_index=255)

            loss = loss1
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr

def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            lateral_map_1 = model(image)
            # output = output['out']
            confmat.update(target.flatten(), lateral_map_1.argmax(1).flatten())
            dice.update(lateral_map_1, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    return confmat, dice.value.item() * 100

def train_one_epoch_ds(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None

    # #the weight of bce and dice
    # loss_weight = [1.,1.]

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            lateral_map_1, lateral_map_2, lateral_map_3, lateral_map_4 = model(image)
            loss1 = criterion(lateral_map_1, target, loss_weight=loss_weight, num_classes=num_classes, ignore_index=255)
            loss2 = criterion(lateral_map_2, target, loss_weight=loss_weight, num_classes=num_classes, ignore_index=255)
            loss3 = criterion(lateral_map_3, target, loss_weight=loss_weight, num_classes=num_classes, ignore_index=255)
            loss4 = criterion(lateral_map_4, target, loss_weight=loss_weight, num_classes=num_classes, ignore_index=255)
            loss = loss1+loss2+loss3+loss4
            # losse_edg = dice_coeff(torch.sigmoid(edge), torch.unsqueeze(target, dim=1).float(), ignore_index=255.)
            # losse_obj = dice_coeff(torch.sigmoid(obj), torch.unsqueeze(target, dim=1).float(), ignore_index=255)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr

def evaluate_ds(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            lateral_map_1, lateral_map_2, lateral_map_3, lateral_map_4 = model(image)
            # output = output['out']
            confmat.update(target.flatten(), lateral_map_1.argmax(1).flatten())
            dice.update(lateral_map_1, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    return confmat, dice.value.item() * 100

def create_lr_scheduler_poly(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=10,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def create_lr_scheduler_cos(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=10,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
