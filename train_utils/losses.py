import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()
    sets_sum = i_flat.sum() + t_flat.sum()
    if sets_sum == 0:
        sets_sum = 2 * intersection

    return 1 - ((2. * intersection + smooth) / ( sets_sum + smooth))


def train_loss(prediction, target, loss_weight=[1.,1.], ignore_index: int = 255):
    # Average of Dice coefficient for all batches, or for a single mask
    # 计算一个batch中所有图片的平均损失近似为batch损失
    d = 0.
    batch_size = prediction.shape[0]
    if prediction.shape[1] == 1:
        prediction = torch.squeeze(prediction)
        target = torch.squeeze(target)
    for i in range(batch_size):
        x_i = prediction[i].reshape(-1)
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]


        bce = F.binary_cross_entropy_with_logits(x_i, t_i.float())
        x_i = F.sigmoid(x_i)
        dice = dice_loss(x_i, t_i)

        loss = bce * loss_weight[0] + dice * loss_weight[1]
        d += loss

    return d / batch_size




# def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
#     """build target for dice coefficient"""
#     dice_target = target.clone()
#     if ignore_index >= 0:
#         ignore_mask = torch.eq(target, ignore_index)#计算像素值为255的位置
#         dice_target[ignore_mask] = 0
#         # [N, H, W] -> [N, H, W, C]
#         #将GT转为针对每一个类别的one-hot编码。Channel-0为背景GT，Channel-1为前景GT
#         dice_target = nn.functional.one_hot(dice_target, num_classes).float()
#         dice_target[ignore_mask] = ignore_index#将one-hot中原来255的像素位置还原
#     else:
#         dice_target = nn.functional.one_hot(dice_target, num_classes).float()
#
#     return dice_target.permute(0, 3, 1, 2)# [N, H, W, C] -> [N, C, H, W]
#
#
def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # 计算一个batch中所有图片某个类别的dice_coefficient
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size
#
#
# def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
#     """Average of Dice coefficient for all classes"""
#     dice = 0.
#     for channel in range(x.shape[1]):
#         dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)
#
#     return dice / x.shape[1]
#
#
# def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
#     # Dice loss (objective to minimize) between 0 and 1
#     x = nn.functional.softmax(x, dim=1)
#     fn = multiclass_dice_coeff if multiclass else dice_coeff
#     return 1 - fn(x, target, ignore_index=ignore_index)
