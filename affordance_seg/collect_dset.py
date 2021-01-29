# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
# crashes sometimes without this for some reason
torch.multiprocessing.set_sharing_strategy('file_system')

import json
import random
import numpy as np
import torch.multiprocessing as mp
import os
import tqdm
import shutil
import collections

from .args import get_args
from rl.common.utils import logger
from rl.common.env_utils import construct_envs, get_env_class
from interaction_exploration.run import RGBTrainer
from interaction_exploration.config import get_config

class CollectAffTrainer(RGBTrainer):

    def __init__(self, config, vis_encoder):
        super().__init__(config, vis_encoder)
        self.episodes = {}

    def print_stats(self):
        if len(self.episodes)==0:
            return 

        total_frames = sum([episode['N_frames'] for episode in self.episodes.values()])
        episode_count = collections.Counter([key[0] for key in self.episodes])

        print ('-'*20)
        print (f'Total frames: {total_frames}')

        print ('-'*20)
        print ('Episode distribution:')
        for episode, count in sorted(episode_count.items(), key=lambda x: int(x[0].split('FloorPlan')[1])):
            print (episode, count)

        print ('-'*20)

    def save_episode(self, observations):

        # torch.Size([17, 3, 80, 80]) torch.Size([17, 7, 2, 80, 80]) torch.Size([17, 5]), info
        frames, masks, poses, info = observations

        # keep episodes only where annotations are present
        neg_score = (masks[:, :, 0]).sum(2).sum(2) # (N, 7)
        pos_score = (masks[:, :, 1]).sum(2).sum(2) # (N, 7)
        scores = (pos_score*neg_score).sum(1) # (N, )
        
        keep_idx = scores>0
        if keep_idx.sum()==0:
            return False

        frames = frames[keep_idx]
        masks = masks[keep_idx]
        poses = poses[keep_idx]
        episode = {'frames':frames, 'masks':masks, 'poses':poses, 'info':info}

        scene, episode_id = episode['info']['scene'], episode['info']['episode']
        out_dir = f'{self.config.OUT_DIR}/episodes/'

        filename = f'{scene}_{episode_id}_data.npz'
        np.savez_compressed(os.path.join(out_dir, filename),
                            frames=episode['frames'],
                            masks=episode['masks'],
                            poses=episode['poses'],
                            )
        torch.save({'info': episode['info'],
                    }, os.path.join(out_dir, f'{scene}_{episode_id}_info.pth'))

        # log information for dataset stats
        key = (info['scene'], info['episode'])
        log_episode = {'N_frames':frames.shape[0]}
        self.episodes[key] = log_episode

        print (f'Saved episode {scene} {episode_id} to {out_dir}')
        return True

    # clone of eval() / _eval_checkpoint() to save images/masks
    def collect_affordance_episodes(self):

        os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)

        # add test episode information to config
        test_episodes = json.load(open(config.EVAL.DATASET))
        self.config.defrost()
        self.config.ENV.TEST_EPISODES = test_episodes
        self.config.ENV.TEST_EPISODE_COUNT = len(test_episodes)
        self.config.freeze()

        # [!!] Load checkpoint, create dir to save rollouts to, and copy checkpoint for reference
        checkpoint_path = self.config.LOAD
        os.makedirs(f'{self.config.OUT_DIR}/episodes/', exist_ok=True)
        shutil.copy(checkpoint_path, f'{self.config.OUT_DIR}/{os.path.basename(checkpoint_path)}')

        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        ppo_cfg = self.config.RL.PPO

        logger.info(f"env config: {self.config}")
        self.envs = construct_envs(self.config, get_env_class(self.config.ENV.ENV_NAME))
        self._setup_actor_critic_agent(ppo_cfg)

        # [!!] Log extra stuff
        logger.info(checkpoint_path)
        logger.info(f"num_steps: {self.config.ENV.NUM_STEPS}")

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = self.batch_obs(observations, self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]

        pbar = tqdm.tqdm()
        self.actor_critic.eval()

        iteration = 0
        while (
            len(stats_episodes) < self.config.ENV.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):

            # [!!] Show more fine-grained progress. THOR is slow!
            pbar.update()

            # [!!] Show episodes collected so far
            if iteration%self.config.ENV.NUM_STEPS == 0:
                print (f'Iter: {iteration}')
                self.print_stats()
            iteration += 1 

            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)

            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = self.batch_obs(observations, self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)

            current_episode_reward += rewards

            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):

                if (
                    next_episodes[i]['scene_id'],
                    next_episodes[i]['episode_id'],
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:

                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0

                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i]['scene_id'],
                            current_episodes[i]['episode_id'],
                        )
                    ] = episode_stats

                    # [!!] save episode data 
                    self.save_episode(infos[i]['traj_masks'])

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )


        self.envs.close()


if __name__=='__main__':

    from interaction_exploration.models.policy import RGBCNN

    mp.set_start_method('spawn')

    args = get_args()
    config = get_config(args.config, opts=args.opts)

    random.seed(config.SEED)
    np.random.seed(config.SEED)

    # Collect episodes using RGB baseline   
    trainer = CollectAffTrainer(config, vis_encoder=RGBCNN)
    trainer.collect_affordance_episodes()
  