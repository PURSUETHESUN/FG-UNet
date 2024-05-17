import os

import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, root: str, flag: str, transforms=None):
        super(DriveDataset, self).__init__()
        # self.flag = "train" if train else "test"
        data_root = os.path.join(root, flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith((".jpg",".png"))]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        # self.mask = [os.path.join(data_root, "mask", i.split('.')[0] + '_segmentation.png') for i in img_names]
        self.mask = [os.path.join(data_root, "mask", i) for i in img_names]
        # check files
        for i in self.mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        # self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
        #                  for i in img_names]
        # # check files
        # for i in self.roi_mask:
        #     if os.path.exists(i) is False:
        #         raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        mask = Image.open(self.mask[idx]).convert('L')
        #前景变1，背景为0。如果mask是调色板格式，不需要除以255
        mask = np.array(mask) / 255
        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

if __name__ == '__main__':
    train_dataset = DriveDataset("../data_set/CVC_ClinicDB", flag='train')
    print(f'train: {len(train_dataset)}')

    val_dataset = DriveDataset("../data_set/CVC_ClinicDB", flag='val')
    print(f'val: {len(val_dataset)}')

    # test_dataset = DriveDataset("../data_set/ISIC2018", flag='test')
    # print(f'test: {len(test_dataset)}')

    #获取第一个images和mask
    # i, t = train_dataset[0]
    # i, t = val_dataset[0]
    # i,t = test_dataset[0]
    # # 创建包含两个子图的图像窗口
    # fig, axes = plt.subplots(1, 2)
    # # 在第一个子图中显示第一幅图像
    # axes[0].imshow(i)
    # axes[0].set_title("Image")
    # # 在第二个子图中显示第二幅图像
    # axes[1].imshow(t.convert('L'),cmap='gray')
    # axes[1].set_title("mask")
    # # 调整子图之间的间距
    # plt.tight_layout()
    # # 显示图像窗口
    # plt.show()