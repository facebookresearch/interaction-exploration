# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
import shutil
from gym import spaces

from rl.common.utils import batch_obs
from rl.common.rollout_storage import RolloutStorage
from .models.policy import PolicyNetwork
from .viz_trainer import VizTrainer

class RGBTrainer(VizTrainer):

    def __init__(self, config, vis_encoder):
        super().__init__(config)
        self.vis_encoder = vis_encoder
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.img_sz = 80

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    def transform(self, tensor): # (B, H, W, 3) [0, 255]
        rgb = tensor.permute(0, 3, 1, 2)/255 # (B, 3, H, W) [0, 1]
        rgb = F.interpolate(rgb, self.img_sz, mode='bilinear', align_corners=True)
        rgb = (rgb - self.mean.to(rgb.device))/self.std.to(rgb.device)
        return rgb
        
    def batch_obs(self, observations, device=None):
        batch = batch_obs(observations, device) # rgb: (32, 300, 300, 3) [0, 255]
        rgb = self.transform(batch['rgb'])
        batch['rgb'] = rgb
        return batch

    def _init_actor_critic_model(self, ppo_cfg):
        actor_critic = PolicyNetwork(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            vis_encoder=self.vis_encoder,
        )

        if ppo_cfg.policy_wts != "":
            wts = torch.load(ppo_cfg.policy_wts)['state_dict']
            wts = {k.replace('actor_critic.',''):v for k,v in wts.items()}
            actor_critic.load_state_dict(wts, strict=True)
            print ('Initialized policy network with pretrained weights')

        return actor_critic

    @classmethod
    def _extract_scalars_from_info(cls, info):
        return {}

    # use to add extra observation sensors
    def augment_obs_space(self, obs_space):
        return obs_space

    def create_rollout_storage(self, ppo_cfg):
        obs_space = copy.deepcopy(self.envs.observation_spaces[0])
        obs_space = self.augment_obs_space(obs_space)

        rollouts = RolloutStorage(
            ppo_cfg.hidden_size,
            self.envs.num_envs,
            obs_space,
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
        )
        return rollouts

from affordance_seg.unet import UNet
class BeaconTrainer(RGBTrainer):

    def __init__(self, config, vis_encoder):
        super().__init__(config, vis_encoder)
        self.unet = UNet(config)

        weights = torch.load(config.MODEL.BEACON_MODEL, map_location='cpu')['state_dict']
        self.unet.load_state_dict(weights)
        self.unet.eval().to(self.device)
        print ('UNet checkpoint loaded')

        os.makedirs(os.path.join(config.CHECKPOINT_FOLDER, 'unet/'), exist_ok=True)
        if not os.path.exists(os.path.join(config.CHECKPOINT_FOLDER, 'unet/', os.path.basename(config.MODEL.BEACON_MODEL))):
            shutil.copy(config.MODEL.BEACON_MODEL, os.path.join(config.CHECKPOINT_FOLDER, 'unet/'))
            print ('Saved UNet checkpoint to cv_dir')

    def batch_obs(self, observations, device=None):
        batch = super().batch_obs(observations, device)
        with torch.no_grad():
            imgs = batch['rgb'].to(self.device)
            aux = self.unet.get_processed_affordances(imgs)
            aux = (aux-self.config.DATA.AUX_MEAN)/self.config.DATA.AUX_STD
        batch['aux'] = aux
        return batch

    def augment_obs_space(self, obs_space):
        obs_space.spaces['aux'] = spaces.Box(-np.inf, np.inf, (7, 80, 80))
        return obs_space

from .models.policy import RandomPolicy
class RandomTrainer(RGBTrainer):
    def __init__(self, config, vis_encoder):
        super().__init__(config, vis_encoder)

    def load_checkpoint(self, checkpoint_path, *args, **kwargs):
        return None

    def _init_actor_critic_model(self, ppo_cfg):
        actor_critic = RandomPolicy(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            vis_encoder=self.vis_encoder,
        )
        return actor_critic

from .models.mlnet import MLNet
class SaliencyTrainer(RGBTrainer):

    def __init__(self, config, vis_encoder):
        super().__init__(config, vis_encoder)
        prior_size = (3, 3) # 240 x 240 image
        self.mlnet = MLNet(config, prior_size)
        self.mlnet.eval().to(self.device)

    def batch_obs(self, observations, device=None):
        batch = super().batch_obs(observations, device)
        sz = batch['rgb'].shape[-1]
        imgs = F.interpolate(batch['rgb'], (240, 240), mode='bilinear', align_corners=True)
        imgs = imgs.to(self.device)
        with torch.no_grad():
            hmaps = self.mlnet(imgs)
            hmaps = F.interpolate(hmaps, (sz, sz), mode='bilinear', align_corners=True)
            norm, _ = hmaps.view(hmaps.shape[0], -1).max(1)
            hmaps = hmaps/(norm.view(hmaps.shape[0], 1, 1, 1) + 1e-12)
        batch['aux'] = (hmaps-self.config.DATA.AUX_MEAN)/self.config.DATA.AUX_STD
        return batch

    def augment_obs_space(self, obs_space):
        obs_space.spaces['aux'] = spaces.Box(-np.inf, np.inf, (1, 80, 80))
        return obs_space

