# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import tqdm
import json
import os

rs = np.random.RandomState(8275)

test_episodes = []
K = 16 # episodes per environment
for scene in tqdm.tqdm(range(1, 5+1)): # Evaluate for all. Split later
    episodes = rs.choice(10000000, K, replace=False)
    for episode in episodes.tolist():
        test_episodes.append((f'FloorPlan{scene}', episode))

os.makedirs('interaction_exploration/data/', exist_ok=True)
json.dump(test_episodes, open(f'interaction_exploration/data/test_episodes_K_{K}.json', 'w'), indent=2)



