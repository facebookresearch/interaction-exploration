# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import tqdm
import json
import os
import collections

rs = np.random.RandomState(8275)

K = 256 # episodes per environment
splits = 4 # separate out to run on individual GPUs
episodes = collections.defaultdict(list)

count = 0
for scene in tqdm.tqdm(range(6, 30+1)): 
    episode_ids = rs.choice(10000000, K, replace=False)
    for ep_id in episode_ids.tolist():
        episodes[count%splits].append((f'FloorPlan{scene}', ep_id))
        count += 1

os.makedirs('affordance_seg/data/episode_splits/', exist_ok=True)
for split in episodes:
	json.dump(episodes[split], open(f'affordance_seg/data/episode_splits/episodes_K_{K}_split_{split}.json', 'w'), indent=2)
