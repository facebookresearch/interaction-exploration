# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import sys


def blend(img1, img2, alpha=0.4):
    img1 = transforms.ToPILImage()(img1).convert("RGBA")
    img2 = transforms.ToPILImage()(img2).convert("RGBA")
    img3 = Image.blend(img1, img2, alpha=alpha).convert('RGB')
    return transforms.ToTensor()(img3)

# Apply .cuda() to every element in the batch
def batch_cuda(batch):
    _batch = {}
    for k,v in batch.items():
        if type(v)==torch.Tensor:
            v = v.cuda()
        elif type(v)==list and type(v[0])==torch.Tensor:
            v = [v.cuda() for v in v]
        _batch.update({k:v})

    return _batch

def load_img(fl):
    return Image.open(fl).convert('RGB')

def unnormalize(tensor):
    mean, std = default_mean_std()
    u_tensor = tensor.clone()

    def _unnorm(t):
        for c in range(3):
            t[c].mul_(std[c]).add_(mean[c])

    if u_tensor.dim()==4:
        [_unnorm(t) for t in u_tensor]
    else:
        _unnorm(u_tensor)
    
    return u_tensor

def default_mean_std():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return mean, std

def default_transform(split):
    mean, std = default_mean_std()

    if split=='train':
        transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])


    elif split=='val':
        transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
            ])

    return transform


def show_wait(img, T=0, win='image', sz=None, save=None):

    shape = img.shape
    img = transforms.ToPILImage()(img)
    if sz is not None:
        H_new = int(sz/shape[2]*shape[1])
        img = img.resize((sz, H_new))

    open_cv_image = np.array(img) 
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    if save is not None:
        cv2.imwrite(save, open_cv_image)
        return

    cv2.imshow(win, open_cv_image)
    inp = cv2.waitKey(T)
    if inp==27:
        cv2.destroyAllWindows()
        sys.exit(0)