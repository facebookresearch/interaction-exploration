# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import yacs.config
import os

# from habitat.config import Config as CN # type: ignore

# Default Habitat config node
class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)


CN = Config

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 1
_C.NUM_PROCESSES = 16
_C.X_DISPLAY = ":0"
_C.DEBUG = False

_C.CUDA = True
_C.CUDA_DETERMINISTIC = False
_C.TORCH_GPU_ID = 0 

_C.CHECKPOINT_INTERVAL = 10
_C.CHECKPOINT_FOLDER = "cv/tmp"

_C.EVAL_CKPT_NUMBER = -1
_C.LOG_FILE = "run.log"
_C.LOG_INTERVAL = 1
_C.TENSORBOARD_DIR = "tb/"

_C.MODE = "train"
_C.LOAD = None

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.AUX_MEAN = None
_C.DATA.AUX_STD = None

# -----------------------------------------------------------------------------
# DOWNSTREAM TASK CONFIG
# -----------------------------------------------------------------------------
_C.TASK_CONFIG = CN()
_C.TASK_CONFIG.TASK = None

# -----------------------------------------------------------------------------
# MISC
# -----------------------------------------------------------------------------
_C.OUT_DIR = "" # Where to store affordance data?

# -----------------------------------------------------------------------------
# EVAL
# -----------------------------------------------------------------------------
_C.EVAL = CN()
_C.EVAL.DATASET = 'interaction_exploration/data/test_episodes_K_16.json'

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TRAINER = "BaseTrainer"
_C.MODEL.ENCODER = "RGBCNN"
_C.MODEL.BEACON_MODEL = ""

# -----------------------------------------------------------------------------
# ENVIRONMENT
# -----------------------------------------------------------------------------
_C.ENV = CN()
_C.ENV.NUM_ENV_STEPS = 1000000
_C.ENV.NUM_STEPS = 256
_C.ENV.ENV_NAME = "ThorInteractionCount-v0"
#_C.ENV.LOCAL_EXE = "/path/to/thor-Linux64-202003231453/thor-Linux64-202003231453" # 2.3.8
_C.ENV.LOCAL_EXE = None # If multiple versions exist, specific path as above
_C.ENV.OBS_SZ = 80
_C.ENV.ROT_SIZE_X = 15
_C.ENV.ROT_SIZE_Y = 30
_C.ENV.FRAME_SIZE = 300
_C.ENV.NGRID = 5

# -----------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION (PPO)
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.PPO = CN()
_C.RL.PPO.clip_param = 0.2
_C.RL.PPO.ppo_epoch = 4
_C.RL.PPO.num_mini_batch = 2
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.entropy_coef = 0.01
_C.RL.PPO.lr = 2e-4
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.5
_C.RL.PPO.num_steps = 256
_C.RL.PPO.use_gae = True
_C.RL.PPO.use_linear_lr_decay = False
_C.RL.PPO.use_linear_clip_decay = False
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
_C.RL.PPO.reward_window_size = 10
_C.RL.PPO.use_normalized_advantage = False
_C.RL.PPO.gae_lambda = 0.95
_C.RL.PPO.hidden_size = 512
_C.RL.PPO.policy_wts = ""

# -----------------------------------------------------------------------------


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    # [!!] Add some extra opts
    config.LOG_FILE = os.path.join(config.CHECKPOINT_FOLDER, 'run.log')
    config.TENSORBOARD_DIR = os.path.join(config.CHECKPOINT_FOLDER, 'tb/')


    config.freeze()
    return config
