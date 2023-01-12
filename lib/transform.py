# import os
# import PIL
# 
# from torchvision import datasets, transforms
# from timm.data import create_transform
# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# 
# mean_ilsvrc12 = [0.485, 0.456, 0.406]
# std_ilsvrc12 = [0.229, 0.224, 0.225]
# mean_inat19 = [0.454, 0.474, 0.367]
# std_inat19 = [0.237, 0.230, 0.249]
# 
# normalize_tfs_ilsvrc12 = transforms.Normalize(mean=mean_ilsvrc12, std=std_ilsvrc12)
# normalize_tfs_inat19 = transforms.Normalize(mean=mean_inat19, std=std_inat19)
# normalize_tfs_dict = {
#     "tiered-imagenet-84": normalize_tfs_ilsvrc12,
#     "tiered-imagenet-224": normalize_tfs_ilsvrc12,
#     "ilsvrc12": normalize_tfs_ilsvrc12,
#     "inaturalist19-84": normalize_tfs_inat19,
#     "inaturalist19-224": normalize_tfs_inat19,
# }
# 
# 
# def train_transforms(img_resolution, dataset, augment=True, normalize=True):
#     if augment and normalize:
#         return transforms.Compose(
#             [
#                 # extract random crops and resize to img_resolution
#                 transforms.RandomResizedCrop(img_resolution),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 normalize_tfs_dict[dataset],
#             ]
#         )
#     elif not augment and normalize:
#         return transforms.Compose([transforms.ToTensor(), normalize_tfs_dict[dataset]])
#     elif augment and not normalize:
#         return transforms.Compose([transforms.RandomResizedCrop(img_resolution), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
#     else:
#         return transforms.Compose([transforms.ToTensor()])
# 
# 
# def val_transforms(dataset, normalize=True, resize=None, crop=None):
#     trsfs = []
#     
#     if resize:
#         trsfs.append(transforms.Resize((resize, resize)))
# 
#     if crop:
#         trsfs.append(transforms.CenterCrop(crop))
# 
#     if normalize:
#         trsfs.extend([transforms.ToTensor(), normalize_tfs_dict[dataset]])
#     else:
#         trsfs.append([*transforms.ToTensor()])
# 
#     return transforms.Compose(trsfs)
# 
# def build_transform(is_train, args):
#     mean = IMAGENET_DEFAULT_MEAN
#     std = IMAGENET_DEFAULT_STD
#     # train transform
#     if is_train:
#         # this should always dispatch to transforms_imagenet_train
#         transform = create_transform(
#             input_size=args.input_size,
#             is_training=True,
#             color_jitter=args.color_jitter,
#             auto_augment=args.aa,
#             interpolation='bicubic',
#             re_prob=args.reprob,
#             re_mode=args.remode,
#             re_count=args.recount,
#             mean=mean,
#             std=std,
#         )
#         return transform
# 
#     # eval transform
#     t = []
#     if args.input_size <= 224:
#         crop_pct = 224 / 256
#     else:
#         crop_pct = 1.0
#     size = int(args.input_size / crop_pct)
#     t.append(
#         transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
#     )
#     t.append(transforms.CenterCrop(args.input_size))
# 
#     t.append(transforms.ToTensor())
#     t.append(transforms.Normalize(mean, std))
#     return transforms.Compose(t)

import torch
import torch.nn.functional as F
import random

class transform:
    def __init__(self, flip=True, r_crop=True, g_noise=True):
        self.flip = flip
        self.r_crop = r_crop
        self.g_noise = g_noise
        print("holizontal flip : {}, random crop : {}, gaussian noise : {}".format(
            self.flip, self.r_crop, self.g_noise
        ))

    def __call__(self, x):
        if self.flip and random.random() > 0.5:
            x = x.flip(-1)
        if self.r_crop:
            h, w = x.shape[-2:]
            x = F.pad(x, [2,2,2,2], mode="reflect")
            l, t = random.randint(0, 4), random.randint(0,4)
            x = x[:,:,t:t+h,l:l+w]
        if self.g_noise:
            n = torch.randn_like(x) * 0.15
            x = n + x
        return x
