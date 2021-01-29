# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch.multiprocessing as mp
import random
import numpy as np

from .config import get_config
from .args import get_args
from .trainer import *
from .models.policy import * 

def get_trainer(config):
    Trainer = globals()[config.MODEL.TRAINER]
    encoder = globals()[config.MODEL.ENCODER]
    trainer = Trainer(config, vis_encoder=encoder)
    print ('Using trainer:', Trainer, ' | Encoder:', encoder)
    return trainer

if __name__=='__main__':

    mp.set_start_method('spawn')

    args = get_args()
    config = get_config(args.config, opts=args.opts)

    random.seed(config.SEED)
    np.random.seed(config.SEED)

    trainer = get_trainer(config)

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'eval':
        trainer.eval()
    elif args.mode == 'enjoy':
        trainer.enjoy()














