# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import pytorch_lightning as pl
from einops import rearrange
import segmentation_models_pytorch as smp

from .dataset import AffordanceDataset

class UNet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.nout = 4
        self.unet = smp.Unet('resnet18', encoder_weights='imagenet', classes=7*self.nout, encoder_depth=3, decoder_channels=(128, 64, 32))
        for name, param in self.unet.named_parameters():
            if name.startswith('encoder'):
                param.requires_grad = False

    def forward(self, x):
        self.unet.encoder.eval()
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)
        masks = self.unet.segmentation_head(decoder_output)
        return masks


    def get_preds(self, x, resize=None):
        out = self.forward(x)
        C = out.shape[1]

        if resize is not None:
            out = F.interpolate(out, resize, mode='bilinear', align_corners=True)
        out = rearrange(out, 'b (c1 c2) h w -> b c1 c2 h w', c1=self.nout, c2=C//self.nout) # (B, 3, 7, 80, 80)
        preds_act = out[:, :2]
        preds_fs = out[:, 2:]
        preds = {'fs': preds_fs, 'act':preds_act}

        return preds

    def get_processed_affordances(self, x):

        preds = self.get_preds(x)
        pred_act, pred_fs = preds['act'], preds['fs'] # (B, 2, 7, H, W), (B, 2, 7, H, W)

        # affordibility
        act_probs = [nn.Softmax2d()(pred_act[:, :, ch]) for ch in range(7)]
        act_probs = torch.stack(act_probs, 2) # (B, 2, 7, 300, 300)

        # interactibility
        fs_probs = [nn.Softmax2d()(pred_fs[:, :, ch]) for ch in range(7)]
        fs_probs = torch.stack(fs_probs, 2) # (B, 2, 7, 300, 300)

        pred_idx = act_probs[:, 1] * fs_probs[:, 0]

        return pred_idx

    def mask_loss(self, masks, preds):
        preds_fs, preds_act = preds['fs'], preds['act']

        bin_wt = self.class_wts.to(preds_fs.device) # (2, 7)
        loss_act = []
        for ch in range(7):
            loss =  F.cross_entropy(preds_act[:, :, ch], masks[:, ch].long(), ignore_index=2, weight=bin_wt[:, ch]) 
            loss_act.append(loss)
        loss_act = sum(loss_act)/len(loss_act)

        loss_fs = F.cross_entropy(preds_fs, (masks==2).long())
        loss = loss_act + loss_fs 

        return loss

    def training_step(self, batch, idx):
        frames, masks = batch['frame'], batch['mask']
        preds = self.get_preds(frames)
        loss = self.mask_loss(masks, preds)
        logs = {'train_loss': loss}        
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, idx):
        frames, masks = batch['frame'], batch['mask']
        preds = self.get_preds(frames)
        loss = self.mask_loss(masks, preds)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.cat([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        print (f'AVERAGE VAL LOSS = {avg_loss.item()}')
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def prepare_data(self):
        dset = AffordanceDataset(out_sz=80)
        dset.load_entries(self.args.data_dir)
        self.trainset = copy.deepcopy(dset).set_mode('train')
        self.valset = copy.deepcopy(dset).set_mode('val')
        self.class_wts = self.trainset.get_class_wts()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.workers)

    def configure_optimizers(self):
        params = list(filter(lambda p: p.requires_grad, self.parameters()))
        print('%d params to optimize'%len(params))
        optimizer = torch.optim.Adam(params, self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.decay_after], gamma=0.1)
        return [optimizer], [scheduler]