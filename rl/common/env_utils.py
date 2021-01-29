# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import glob
import os
import numpy as np

# [!!] Remove habitat imports
# import habitat
# from habitat import Config, Env, RLEnv, VectorEnv, make_dataset
from .vector_env import VectorEnv

# [++] Get envs.thor* classes
def get_env_class(env_name):
    env_name = env_name.split('-')[0]
    for fl in glob.glob('envs/*.py'):
        module = os.path.basename(fl)[:-3] # remove dir and extension
        m = importlib.import_module(f'.{module}', package='envs')
        if hasattr(m, env_name):
            return getattr(m, env_name)

    return None

def make_env_fn(config, env_class, rank):
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.
    Args:
        config: root exp config that has core env config node as well as
            env-specific config node.
        env_class: class type of the env to be created.
        rank: rank of env to be created (for seeding).
    Returns:
        env object created according to specification.
    """
    env = env_class(config=config)
    env.seed(rank)
    return env


def construct_envs(config, env_class):
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.
    Args:
        config: configs that contain num_processes as well as information
        necessary to create individual environments.
        env_class: class type of the envs to be created.
        specified by the trainer
        randseed: use for debugging
    Returns:
        VectorEnv object created according to specification.
    """

    if config.MODE == 'eval':
        test_episodes = [[] for _ in range(config.NUM_PROCESSES)]
        for idx, episode in enumerate(config.ENV.TEST_EPISODES):
            test_episodes[idx%config.NUM_PROCESSES].append(episode)

        # if there are fewer than num_process episodes, reduce num_processes
        test_episodes = [eps for eps in test_episodes if len(eps)>0]
        # duplicate last episode to pause thread
        test_episodes = [eps + [eps[-1]] for eps in test_episodes]

        config.defrost()
        config.NUM_PROCESSES = len(test_episodes)
        config.ENV.TEST_EPISODES = test_episodes
        config.freeze()

    num_processes = config.NUM_PROCESSES
    env_classes = [env_class for _ in range(num_processes)]

    displays = [None]
    if config.X_DISPLAY is not None:
        displays = config.X_DISPLAY.strip(':').split(',')
    displays = displays*(num_processes//len(displays) + 1)

    # create {num_processes} configs, one for each environment
    configs = []
    for i in range(num_processes):
        proc_config = config.clone()
        proc_config.defrost()

        proc_config.X_DISPLAY = displays[i]
        
        # if testing, force the environment to iterate through a fixed set of episodes
        if config.MODE=='eval':
            proc_config.ENV.TEST_EPISODES = config.ENV.TEST_EPISODES[i]

        proc_config.freeze()
        configs.append(proc_config)

    # initialize the ranks for each processes. Ranks are used to seed the env
    if config.MODE=='train':
        ranks = [np.random.randint(1000) for _ in range(num_processes)]
    elif config.MODE=='eval':
        ranks = list(range(num_processes))

    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(zip(configs, env_classes, ranks))
        ),
    )
    return envs

# [!!] Habitat specific
# def make_env_fn(
#     config: Config, env_class: Type[Union[Env, RLEnv]]
# ) -> Union[Env, RLEnv]:
#     r"""Creates an env of type env_class with specified config and rank.
#     This is to be passed in as an argument when creating VectorEnv.

#     Args:
#         config: root exp config that has core env config node as well as
#             env-specific config node.
#         env_class: class type of the env to be created.

#     Returns:
#         env object created according to specification.
#     """
#     dataset = make_dataset(
#         config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET
#     )
#     env = env_class(config=config, dataset=dataset)
#     env.seed(config.TASK_CONFIG.SEED)
#     return env


# def construct_envs(
#     config: Config, env_class: Type[Union[Env, RLEnv]]
# ) -> VectorEnv:
#     r"""Create VectorEnv object with specified config and env class type.
#     To allow better performance, dataset are split into small ones for
#     each individual env, grouped by scenes.

#     Args:
#         config: configs that contain num_processes as well as information
#         necessary to create individual environments.
#         env_class: class type of the envs to be created.

#     Returns:
#         VectorEnv object created according to specification.
#     """

#     num_processes = config.NUM_PROCESSES
#     configs = []
#     env_classes = [env_class for _ in range(num_processes)]
#     dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
#     scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
#     if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
#         scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

#     if num_processes > 1:
#         if len(scenes) == 0:
#             raise RuntimeError(
#                 "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
#             )

#         if len(scenes) < num_processes:
#             raise RuntimeError(
#                 "reduce the number of processes as there "
#                 "aren't enough number of scenes"
#             )

#         random.shuffle(scenes)

#     scene_splits = [[] for _ in range(num_processes)]
#     for idx, scene in enumerate(scenes):
#         scene_splits[idx % len(scene_splits)].append(scene)

#     assert sum(map(len, scene_splits)) == len(scenes)

#     for i in range(num_processes):
#         proc_config = config.clone()
#         proc_config.defrost()

#         task_config = proc_config.TASK_CONFIG
#         task_config.SEED = task_config.SEED + i
#         if len(scenes) > 0:
#             task_config.DATASET.CONTENT_SCENES = scene_splits[i]

#         task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
#             config.SIMULATOR_GPU_ID
#         )

#         task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

#         proc_config.freeze()
#         configs.append(proc_config)

#     envs = habitat.VectorEnv(
#         make_env_fn=make_env_fn,
#         env_fn_args=tuple(tuple(zip(configs, env_classes))),
#     )
#     return envs
