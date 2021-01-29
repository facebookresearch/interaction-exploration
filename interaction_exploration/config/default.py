# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from rl.config import get_config as rl_get_config

def get_config(config_paths, opts):

    config = rl_get_config(config_paths, opts)

    config.defrost()
    config.NUM_UPDATES = int(config.ENV.NUM_ENV_STEPS) // config.ENV.NUM_STEPS // config.NUM_PROCESSES 
    config.freeze()

    return config