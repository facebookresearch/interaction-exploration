# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import to_rgb
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

import rl.ppo.ppo_trainer as ppo_trainer
from rl.common.utils import logger
from rl.common.env_utils import construct_envs, get_env_class
from .utils import util

class VizTrainer(ppo_trainer.PPOTrainer):

    def __init__(self, config):
        super().__init__(config)

    def init_viz(self):
        sz, N = 300, 5
        center = ((sz//N)*(N//2), (sz//N)*(N+1)//2)
        self.bbox = [center[0], center[0], center[1], center[1]]
        self.interactions = set(['take', 'put', 'open', 'close', 'toggle-on', 'toggle-off', 'slice'])
        self.cmap = matplotlib.cm.get_cmap('Blues')
        # self.font = ImageFont.truetype('interaction_exploration/data/FreeSerif.ttf', 24)


    def reset_fig(self):
        plt.clf()
        fig, ax = plt.subplots(frameon=False, figsize=(3, 3))
        ax.patch.set_facecolor('white')
        ax.patch.set_alpha(0.0)
        ax.set_axis_off()
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 300)
        self.ax = ax
        self.canvas = FigureCanvas(fig)

    def add_rectangle(self, tensor):
        img = transforms.ToPILImage()(tensor)
        draw = ImageDraw.Draw(img)
        draw.rectangle(self.bbox,  outline='blue', width=3)
        tensor = transforms.ToTensor()(img)
        return tensor

    def add_banner(self, draw, text, color):
        sz = 300
        draw.rectangle(((0, 0), (sz//2, sz//8)), fill=to_rgb(color)+(128,))
        draw.text((10, 0), text, fill='white', font=self.font)

    # [!!] Clone of eval, except don't update the optimizer, and visualize frames + actions
    def enjoy(self):

        self.init_viz()

        test_episodes = [(f'FloorPlan{np.random.randint(1, 31)}', np.random.randint(10000)) for _ in range(10)]
        self.config.defrost()
        self.config.ENV.TEST_EPISODES = test_episodes
        self.config.ENV.TEST_EPISODE_COUNT = len(test_episodes)
        self.config.NUM_PROCESSES = 1
        self.config.MODE = 'eval'
        self.config.freeze()

        checkpoint_path = self.config.LOAD
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        ppo_cfg = self.config.RL.PPO

        logger.info(f"env config: {self.config}")

        # choose the right env wrapper depending on class
        env_name = 'ThorEnjoyVanilla'
        if self.config.ENV.ENV_NAME in ['ThorObjectCoverage-v0']:
            env_name = 'ThorEnjoyCycler'
        elif self.config.ENV.ENV_NAME in ['ThorNavigationNovelty-v0']:
            env_name = 'ThorEnjoyCyclerFixedView'

        self.envs = construct_envs(self.config, get_env_class(env_name))
        self._setup_actor_critic_agent(ppo_cfg)

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
        stats_episodes = dict()  
        self.actor_critic.eval()
        while True:

            infos = None
            last_pt = None
            self.reset_fig()
            for step in range(ppo_cfg.num_steps):
                
                # ------------------------------------------- #
                action = 'init' if infos is None else infos[0]['action']
                ep_reward = current_episode_reward[0].item()

                print(f'step: {step} | R: {ep_reward} | {action}')

                viz_data = self.envs.call_at(0, 'get_viz_data')
                frame = torch.from_numpy(viz_data['frame']).float().permute(2, 0, 1)/255
                frame = self.add_rectangle(frame)
                topdown = viz_data['topdown']

                if action!='init':
                    traj_pt = list(viz_data['pts'][0])
                    traj_pt[1] = 300 - traj_pt[1]
                    int_pt = (max(min(viz_data['pts'][-1][0], 295), 5), 300-max(min(viz_data['pts'][-1][1], 295), 5))

                    if last_pt is not None:
                        self.ax.plot((last_pt[0], traj_pt[0]), (last_pt[1], traj_pt[1]), color=self.cmap(step/ppo_cfg.num_steps), lw=2)
                    last_pt = traj_pt

                    if viz_data['action'] in self.interactions:
                        if viz_data['reward']>0:
                            plt.plot(int_pt[0], int_pt[1], marker='o', color='Lime', alpha=0.8)
                        else:
                            plt.plot(int_pt[0], int_pt[1], marker='o', color='yellow', alpha=0.05)


                    self.canvas.draw()

                    s, (width, height) = self.canvas.print_to_buffer()
                    # annots = np.frombuffer(s, np.uint8).reshape((height, width, 4))
                    annots = Image.frombytes("RGBA", (width, height), s)
                    draw = ImageDraw.Draw(annots)
                    draw.polygon(viz_data['pts'][:3], fill=(0, 255, 255, 64))

                    topdown = Image.fromarray(topdown, "RGB").convert("RGBA")
                    topdown = Image.alpha_composite(topdown, annots)
                    topdown = np.array(topdown.convert("RGB"))

                topdown = torch.from_numpy(topdown).float().permute(2, 0, 1)/255

                grid = make_grid([frame, topdown], nrow=2)
                util.show_wait(grid, T=1)

                # ------------------------------------------- #

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
                n_envs = self.envs.num_envs
                for i in range(n_envs):
                    # episode ended
                    if not_done_masks[i].item() == 0:
                        episode_stats = dict()
                        episode_stats["reward"] = current_episode_reward[i].item()
                        episode_stats.update(
                            self._extract_scalars_from_info(infos[i])
                        )
                        current_episode_reward[i] = 0
                        stats_episodes[
                            (
                                current_episodes[i]['scene_id'],
                                current_episodes[i]['episode_id'],
                            )
                        ] = episode_stats

            # Log info so far
            num_episodes = len(stats_episodes)
            aggregated_stats = dict()
            for stat_key in next(iter(stats_episodes.values())).keys():
                aggregated_stats[stat_key] = (
                    sum([v[stat_key] for v in stats_episodes.values()])
                    / num_episodes
                )

            for k, v in aggregated_stats.items():
                logger.info(f"Average episode {k}: {v:.4f} ({num_episodes} episodes)")
