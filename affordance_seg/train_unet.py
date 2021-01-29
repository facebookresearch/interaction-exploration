# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pytorch_lightning as pl
from torchvision.utils import make_grid
from pytorch_lightning.callbacks import ModelCheckpoint

from interaction_exploration.utils import util
from .dataset import AffordanceDataset
from .unet import UNet
    
def resize(tensor, sz=300, mode='bilinear'):

    ndim = tensor.dim()
    if ndim==3:
        tensor = tensor.unsqueeze(0)

    kwargs = {'mode':mode}
    if mode=='bilinear':
        kwargs['align_corners'] = True
    tensor = F.interpolate(tensor, (sz, sz), **kwargs)

    if ndim==3:
        tensor = tensor[0]

    return tensor

def train(args):

    model = UNet(args)

    checkpoint_callback = ModelCheckpoint(
        filepath=args.cv_dir+'/{epoch:02d}-{val_loss:.4f}',
        save_top_k=5,
        verbose=True,
    )

    trainer = pl.Trainer(
        checkpoint_callback=checkpoint_callback,
        default_root_dir=args.cv_dir,
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        distributed_backend='dp',
    )

    trainer.fit(model)


class ColorOverlay:
    def __init__(self, sz):
        color_map = {'blue': [19, 90, 212],
                     'green':[7, 212, 0],
                     'orange':[255, 153, 0],
                     'pink':[224, 20, 204]}
        color_map = {color: torch.Tensor(color_map[color]).unsqueeze(1)/255 for color in color_map}
        self.color_map = color_map
        self.viz_mask = torch.zeros(3, sz, sz)


    def make_color_mask(self, pred, uframe, channel_to_color):
        self.viz_mask.zero_()
        for ch in channel_to_color:
            if torch.any(pred[ch]==1):
                self.viz_mask[:, pred[ch]==1] = self.color_map[channel_to_color[ch]]
        out = util.blend(uframe, self.viz_mask)
        return out

def viz(args, sz=300):

    # create dataset
    dset = AffordanceDataset(out_sz=80)
    dset.load_entries(args.data_dir)
    dset.set_mode('val')

    np.random.seed(10)
    np.random.shuffle(dset.data)

    # create model
    net = UNet(args)
    net.load_state_dict(torch.load(args.load)['state_dict'])
    net.cuda().eval()

    # color coded interactions
    interactions = ['take', 'put', 'open' ,'close', 'toggle-on', 'toggle-off', 'slice']
    viz_actions = [('put', 'green'), ('open', 'pink'), ('toggle-on', 'blue'), ('take', 'orange')]
    channel_to_color = {interactions.index(act):color for act, color in viz_actions}
    color_overlay = ColorOverlay(sz)

    for idx, instance in enumerate(dset):

        frame, mask = instance['frame'], instance['mask']

        uframe = util.unnormalize(frame)
        uframe = resize(uframe, sz)
        mask = resize(mask, sz, 'nearest')

        viz_tensors = []

        # GT mask
        out = color_overlay.make_color_mask(mask, uframe, channel_to_color)
        viz_tensors.append(out)

        # Predictions
        with torch.no_grad():
            frame = F.interpolate(frame.unsqueeze(0), 80, mode='bilinear', align_corners=True)[0]
            preds = net.get_preds(frame.cuda().unsqueeze(0), resize=sz)
            preds = {k:v[0].cpu() for k,v in preds.items()}
            preds = {k:resize(v, sz) for k,v in preds.items()}
        pred_idx = preds['act'].argmax(0)

        # probabilities
        pred_act = preds['act'] # (2, 7, H, W)
        act_probs = [nn.Softmax2d()(pred_act[:, ch].unsqueeze(0))[0] for ch in range(7)]
        act_probs = torch.stack(act_probs, 1) # (3, 7, 300, 300)

        pred_fs = preds['fs']
        fs_probs = [nn.Softmax2d()(pred_fs[:, ch].unsqueeze(0))[0] for ch in range(7)]
        fs_probs = torch.stack(fs_probs, 1) # (2, 7, 300, 300)

        # entropy
        act_entropy = (-act_probs * torch.log(act_probs+1e-12)).sum(0) # (7, 300, 300)
        act_entropy_mask = act_entropy > 0.5*np.log(act_probs.shape[0]) # ignore these values

        # entropy masked
        pred = (pred_idx==1) & (~act_entropy_mask)
        out = color_overlay.make_color_mask(pred, uframe, channel_to_color)
        viz_tensors.append(out)

        grid = make_grid(viz_tensors, nrow=len(viz_tensors))
        util.show_wait(grid, T=0)

def mean_stdev_stats(args):

    dset = AffordanceDataset(out_sz=80)
    dset.load_examples(args.data_dir)
    dset.set_mode('train')

    net = UNet(args)
    net.load_state_dict(torch.load(args.load)['state_dict'])
    net.cuda().eval()

    loader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=True)

    means, stds = [], []
    for batch in tqdm.tqdm(loader, total=len(loader)):
        frames, masks = batch['frame'], batch['mask']

        with torch.no_grad():
            preds = net.get_processed_affordances(frames.cuda()) # (32, 7, 80, 80)

        preds = preds.view(preds.shape[0], 7, -1)
        means.append(preds.mean(2))
        stds.append(preds.std(2))
    means = torch.cat(means, 0).mean(0)
    stds = torch.cat(stds, 0).mean(0)
    print (means)
    print (stds)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size for training')
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='Weight decay for SGD')
    parser.add_argument('--cv_dir', default='cv/tmp/',help='Directory for saving checkpoint nets')
    parser.add_argument('--load', default=None)
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--gpus', default=8, type=int)
    parser.add_argument('--print_every', default=10, type=int)
    parser.add_argument('--decay_after', default=35, type=int)
    parser.add_argument('--max_epochs', default=40, type=int)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--train', action ='store_true', default=False)
    parser.add_argument('--viz', action ='store_true', default=False)
    parser.add_argument('--eval', action ='store_true', default=False)
    args = parser.parse_args()


    if not os.path.exists(f'{args.data_dir}/seg_data.npz'):
        dset = AffordanceDataset(out_sz=80)
        dset.populate_dset(f'{args.data_dir}/episodes/')
        dset.save_entries(args.data_dir)

    if args.train:
        train(args)
    elif args.viz:
        viz(args)





