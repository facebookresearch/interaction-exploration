# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--config', default=None, nargs='+', help='The config files to load')
    parser.add_argument('--out-dir', default='cv/tmp', help='Directory to store affordance dset')    
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help="Modify config options from command line")
    args = parser.parse_args()

    # add args into config
    args.opts += ['OUT_DIR', args.out_dir]

    return args
