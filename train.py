import argparse
import os
import time
import datetime
from typing import Union, List

import torch
from models.FG_UNet import FG as build_model
from train_utils import train_one_epoch_ds, evaluate_ds, create_lr_scheduler_poly, create_lr_scheduler_cos
from my_dataset import DriveDataset
import transforms as T

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.75 * base_size)
        max_size = int(1.25 * base_size)

        self.transforms = T.Compose([
            T.RandomResize(min_size,max_size),
            T.RandomCrop(crop_size),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self,base_size, test_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size),
            T.CenterCrop(test_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")
    base_size = 352
    crop_size = 352
    test_size = 352

    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # 用来保存训练以及验证过程中信息
    results_file = "result_model.txt"

    train_dataset = DriveDataset(args.data_path,
                                 flag='train',
                                 transforms=SegmentationPresetTrain(base_size=base_size, crop_size=crop_size))

    val_dataset = DriveDataset(args.data_path,
                               flag='val',
                               transforms=SegmentationPresetEval(base_size = test_size, test_size=test_size))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = build_model(seg_classes=num_classes)#
    model.to(device)
    print("numworks:",num_workers)

    # if args.weights != "":
    #     weights_dict = torch.load("./deeplabv3_resnet50_coco.pth", map_location='cpu')
    #
    #     if num_classes != 2:
    #         # 官方提供的预训练权重是21类(包括背景)
    #         # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
    #         for k in list(weights_dict.keys()):
    #             if "classifier.4" in k:
    #                 del weights_dict[k]
    #
    #     missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    #     if len(missing_keys) != 0 or len(unexpected_keys) != 0:
    #         print("missing_keys: ", missing_keys)
    #         print("unexpected_keys: ", unexpected_keys)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler_poly(optimizer, len(train_loader), args.epochs, warmup=True)

    # import matplotlib.pyplot as plt
    # lr_list = []
    # for _ in range(args.epochs):
    #     for _ in range(len(train_loader)):
    #         lr_scheduler.step()
    #         lr = optimizer.param_groups[0]["lr"]
    #         lr_list.append(lr)
    # plt.plot(range(len(lr_list)), lr_list)
    # plt.show()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_dice = 0.
    best_epoch = 0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch_ds(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat, dice = evaluate_ds(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        # print(val_info)
        # print(f"dice coefficient: {dice:.3f}")
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if best_dice < dice:
            best_dice = dice
            best_epoch = epoch
            torch.save(save_file, "save_weights/" + "best_models.pth")

        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}] " + f"loss: {mean_loss:.4f} " + f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + f"Dice: {dice:.2f}\n" + f"Best_epoch: {best_epoch}\t" + f"Best_Dice: {best_dice :0.2f}\n\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="../datasets/kvasir_seg", help="DRIVE root")#generalize,kvasir_seg
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=100, type=int, metavar="N",
                        help="number of total epochs to train")
    #select device
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    main(args)
