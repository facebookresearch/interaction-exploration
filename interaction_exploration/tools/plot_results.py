# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import glob
import os
import matplotlib.pyplot as plt
import collections
import torch
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cv_dir', default='cv/')
parser.add_argument('--models', nargs='+')
args = parser.parse_args()

test_scenes = set([f'FloorPlan{idx}' for idx in range(1, 6)])

# max interactions by greedy planning based oracle agent
# See tools/compute_max_interactions.py for these values
max_interactons = {'FloorPlan1': 94, 'FloorPlan2': 94, 'FloorPlan3': 69, 'FloorPlan4': 65, 'FloorPlan5': 92}
def parse_episodes(model, episodes_dir, K):
    episodes = list(glob.glob(f'{episodes_dir}/*.pth'))
    episodes = [torch.load(ep) for ep in episodes]
    episodes = [ep for ep in episodes if ep['scene_id'] in test_scenes] # keep only test episodes
    if len(episodes)==0:
        return {}

    # normalize coverage by oracle interactions and organize by scene
    episodes_by_scene = collections.defaultdict(list)
    for episode in episodes:
        N = max_interactons[episode['scene_id']]
        episode['rewards'] = [step['reward']/N for step in episode['stats']['step_info']]
        episodes_by_scene[episode['scene_id']].append(episode)

    episodes = []
    for scene in episodes_by_scene:
        episodes += episodes_by_scene[scene][:K]

    return episodes

def plot_multirun(eval_episodes, label, ax, T=None, show_std=True, **kwargs):

    viz_data = []
    for run in eval_episodes:
        episodes = eval_episodes[run]
        rewards = np.array([episode['rewards'] for episode in episodes]) # (N, T)

        if T is not None:
            rewards = rewards[:, :T]

        x = range(rewards.shape[1])
        rewards = rewards.cumsum(1) # (N, T)
        y = rewards.mean(0) # (N, )
        viz_data.append({'x':x, 'y':y})

        print (f'{label} | {run} ({len(eval_episodes)}) | N={len(episodes)}')

    run_rewards = np.array([eval_episodes['y'] for eval_episodes in viz_data])
    rewards_mean = run_rewards.mean(0) # (N, )
    rewards_err = run_rewards.std(0) # (N, )

    ax.plot(viz_data[0]['x'], rewards_mean, label=label, **kwargs)
    if show_std:
        ax.fill_between(viz_data[0]['x'], rewards_mean-rewards_err, rewards_mean+rewards_err, alpha=0.1, **kwargs)


# K = number of episodes per scene
# T = max time to display (x axis)
def run_eval(models, labels, ax, colors, K=16, T=None):

    eval_episodes = collections.defaultdict(dict)
    for idx, model in enumerate(models):

        runs = [os.path.basename(dirname.strip('/')) for dirname in glob.glob(f'{args.cv_dir}/{model}/run*/')]
        paths = [f'{args.cv_dir}/{model}/{run}/eval/' for run in runs]

        for run, path in zip(runs, paths):
            eval_episodes[model][run] = parse_episodes(model, path, K)

        plot_multirun(eval_episodes[model], labels[idx], ax, T=T, color=colors[idx])
    

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
models = args.models
labels = models
colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#17becf']

run_eval(models, labels, ax, colors)

ax.legend(loc='upper left')
ax.set_xlabel('timesteps')
ax.set_ylabel('Interaction coverage')
plt.show()
