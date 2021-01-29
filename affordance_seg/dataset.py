# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import glob
import os
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import collections
import torchvision.transforms as transforms
import argparse
import itertools
from sklearn.model_selection import train_test_split
from torchvision.utils import make_grid
from einops import rearrange
from joblib import Parallel, delayed

from interaction_exploration.utils import util


class PairedTransform:
    """
    Transform both the image and an associated mask.
    For image: color jitter, horizontal flip, normalize
    For mask: horizontal flip
    """
    
    def __init__(self, split, out_sz):
        self.split = split
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
        self.out_sz = out_sz

    def resize(self, image, mask, sz):
        if sz != image.shape[-1]:
            image = F.interpolate(image.unsqueeze(0), sz, mode='bilinear', align_corners=True)[0]
            mask = F.interpolate(mask.unsqueeze(0).float(), sz, mode='nearest')[0]
        return image, mask

    def train_transform(self, image, mask): # (3, 80, 80), (7, 80, 80)

        # color jitter 
        image = transforms.ToPILImage()(image)
        image = self.color_jitter(image)
        image = transforms.ToTensor()(image)

        # horizontal flip
        if np.random.rand()<0.5:
            image = image.flip(2)
            mask = mask.flip(2)

        # normalize
        image = TF.normalize(image, self.mean, self.std)

        return image, mask

    def val_transform(self, image, mask):
        image, mask = self.resize(image, mask, self.out_sz)
        image = TF.normalize(image, self.mean, self.std)
        return image, mask

    def __call__(self, image, mask):
        if self.split == 'train':
            image, mask = self.train_transform(image, mask)
        elif self.split =='val':
            image, mask = self.val_transform(image, mask)
        return image, mask

class AffordanceDataset:

    def __init__(self, out_sz):
        self.out_sz = out_sz 
        self.interactions = ['take', 'put', 'open', 'close', 'toggle-on', 'toggle-off', 'slice']
        self.min_pix = 100 # threshold pixels to include (100/80*80 = 1.5%) 
        self.entries = []

    # Extract individual training instances from episode data
    # frames: (N, 3, H, W)
    # masks: (N, 7, 2, H, W)
    def add_episode(self, frames, masks, poses, info):
        
        N, ch, _, H, W = masks.shape
        mask_labels_ = torch.zeros(ch, H, W).fill_(2)
        for idx in range(len(frames)):
            
            # only keep instances that have at least K positive pixels labeled in at least 1 channel
            pos_score = masks[idx, :, 1].view(ch, -1).sum(1) # (7,)
            if torch.all(pos_score < self.min_pix):
                continue

            frame, mask, pose = frames[idx], masks[idx], poses[idx]
            mask_labels = mask_labels_.clone()
            mask_labels[mask[:, 0]==1] = 0
            mask_labels[mask[:, 1]==1] = 1

            self.entries.append({'frame': frame, 'mask':mask_labels, 'pose':pose, 'info':info})

    def load_episode(self, fl):

        episode_data = np.load(fl)
        episode_info = torch.load(fl.replace('_data.npz', '_info.pth'))
        episode = {k:torch.from_numpy(v) for k,v in episode_data.items()}

        frames = episode['frames']
        masks = episode['masks']
        poses = episode['poses']
        info = episode_info['info']

        if frames.shape[-1]!=self.out_sz:
            frames = F.interpolate(frames, self.out_sz, mode='bilinear', align_corners=True)
            masks = rearrange(masks, 'b c two h w -> b (c two) h w')
            masks = F.interpolate(masks.float(), self.out_sz, mode='nearest').byte()
            masks = rearrange(masks, 'b (c two) h w -> b c two h w', c=7, two=2)

        return frames, masks, poses, info

    # split train and val sets (80:20)
    def split_entries(self, entries):
        scenes = [f'FloorPlan{idx}' for idx in range(6, 31)]
        train_scenes, val_scenes = train_test_split(scenes, test_size=0.2, random_state=10)
        train_scenes, val_scenes = set(train_scenes), set(val_scenes)     
        train_data = [entry for entry in entries if entry['info']['scene'] in train_scenes]
        val_data = [entry for entry in entries if entry['info']['scene'] in val_scenes]
        return train_data, val_data

    # Extract training data from K episodes, uniformly distributed across scenes
    def populate_dset(self, data_dir, K=2000):
        
        episodes = list(glob.glob(f'{data_dir}/*.npz'))
        N = len(episodes)
        assert N >= K, f'Not enough episodes collected (# episodes={N}, K={K})'

        # split episodes by scene
        episode_by_scene = collections.defaultdict(list)
        for fl in episodes:
            scene, episode_id, _ = os.path.basename(fl).split('_')
            episode_by_scene[scene].append(fl)

        for scene in episode_by_scene:
        	np.random.shuffle(episode_by_scene[scene])

        # round robin over scenes to re-populate the list of episodes
        episodes = []
        for scene in itertools.cycle(episode_by_scene.keys()):
            if len(episode_by_scene[scene])==0:
                continue

            episodes.append(episode_by_scene[scene].pop())
            if len(episodes)==K:
                break

        print (f'Populated dataset with {K} episodes out of {N} episodes')

        # load all episodes and add each frame to the dataset
        episode_data = Parallel(n_jobs=16, verbose=5)(delayed(self.load_episode)(ep) for ep in episodes)
        for frames, masks, poses, info in tqdm.tqdm(episode_data, total=len(episode_data)):
            self.add_episode(frames, masks, poses, info)

        # split data into train and val set
        self.train_data, self.val_data = self.split_entries(self.entries)

        print (f'Populated with {len(self.entries)} entries')

    # set mode (train or val) and print out split statistics
    def set_mode(self, split):
        self.transform = PairedTransform(split, self.out_sz)
        self.data = self.train_data if split=='train' else self.val_data

        count_by_verb = collections.defaultdict(int)
        for entry in self.data:
            pos_labels = entry['mask'].view(entry['mask'].shape[0], -1)
            pos_labels = (pos_labels==1).float().sum(1) # (7,)
            for ch in range(pos_labels.shape[0]):
                if pos_labels[ch].item() > 0:
                    count_by_verb[ch] += 1
        
        for verb in count_by_verb:
            print (self.interactions[verb], count_by_verb[verb])

        print (f'Data: {len(self.data)} entries ({split})') 
        return self

    # generate cross-entropy weights per class to account for class imbalance
    def get_class_wts(self):

        scores = []
        for entry in self.train_data:
            score = entry['mask'].view(entry['mask'].shape[0], -1)
            scores.append(score)
        scores = torch.stack(scores, 0) # (B, 7, 80*80)

        wts = []
        for idx in range(2):
            wt = (scores==idx).float().sum(2).sum(0) # (7, )
            wts.append(wt)
        wts = torch.stack(wts, 0) # (2, 7)
        wts = wts.sum(0, keepdim=True)/wts 
        wts = torch.clamp(wts/wts.min(0, keepdim=True)[0], 0, 20)

        print ('Weights:', wts)
    
        return wts

    def save_entries(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)

        frames = torch.stack([entry['frame'] for entry in self.entries], 0)
        masks = torch.stack([entry['mask'] for entry in self.entries], 0)
        poses = torch.stack([entry['pose'] for entry in self.entries], 0)
        infos = [entry['info'] for entry in self.entries]

        np.savez_compressed(os.path.join(out_dir, 'seg_data.npz'),
                            frames=frames.numpy(),
                            masks=masks.numpy(),
                            poses=poses.numpy())
        torch.save({'infos':infos}, os.path.join(out_dir, 'seg_data_info.pth'))
        print ('Entries saved to', out_dir)

    def load_entries(self, data_dir):
        seg_data_info = torch.load(os.path.join(data_dir, 'seg_data_info.pth'))
        seg_data = np.load(os.path.join(data_dir, 'seg_data.npz'))

        frames = torch.from_numpy(seg_data['frames'])
        masks = torch.from_numpy(seg_data['masks'])
        poses = torch.from_numpy(seg_data['poses'])
        infos = seg_data_info['infos']

        for idx, (frame, mask, pose, info) in enumerate(zip(frames, masks, poses, infos)):
            self.entries.append({'frame':frame, 'mask':mask, 'pose':pose, 'info':info})
        self.train_data, self.val_data = self.split_entries(self.entries)
        print ('Entries loaded from', data_dir)

    def __getitem__(self, index):
        entry = self.data[index]
        frame, mask = self.transform(entry['frame'], entry['mask'])
        return {'frame':frame, 'mask':mask}

    def __len__(self):
        return len(self.data)


def viz(args, sz=150):

    dset = AffordanceDataset(80)
    dset.load_entries(args.data_dir)
    dset.set_mode('val')
    np.random.shuffle(dset.data)

    for instance in dset:

        uframe = util.unnormalize(instance['frame'])
        mask = instance['mask']

        uframe = F.interpolate(uframe.unsqueeze(0), sz, mode='bilinear', align_corners=True)[0]
        mask = F.interpolate(mask.unsqueeze(0).float(), sz, mode='nearest')[0].byte()

        mask_labels = torch.zeros(3, sz, sz)
        viz_tensors = []
        for ch in range(7):
            mask_labels.zero_()
            mask_labels[0][mask[ch]==0] = 1
            mask_labels[1][mask[ch]==1] = 1
            out = util.blend(uframe, mask_labels)
            viz_tensors.append(out)

        grid = make_grid(viz_tensors, nrow=len(viz_tensors))
        util.show_wait(grid, T=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=None)
    args = parser.parse_args()
    viz(args)
