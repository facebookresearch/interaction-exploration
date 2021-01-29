# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import torch
import torch.nn as nn

from rl.models.policy import Policy, BaseEncoder, Flatten
from rl.models.rnn_state_encoder import RNNStateEncoder


class SimpleCNN(nn.Module):
    
    def __init__(self, observation_space, hidden_size, **kwargs):
        super().__init__()

        self.cnn = nn.Sequential(
                    nn.Conv2d(kwargs['in_channels'], 32, kernel_size=8, stride=4),
                    nn.ReLU(True),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2),
                    nn.ReLU(True),
                    nn.Conv2d(64, 32, kernel_size=3, stride=1),
                    Flatten(),
                    nn.Linear(32*6*6, hidden_size),
                    nn.ReLU(True),
                    )
        self.layer_init()

    def layer_init(self):
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

class RGBCNN(SimpleCNN):

    def __init__(self, observation_space, hidden_size, **kwargs):
        super().__init__(observation_space, hidden_size, in_channels=3)

    def forward(self, observations):
        inputs = observations['rgb']
        output = self.cnn(inputs)
        return output

class TwoStreamNetwork(nn.Module):
    def __init__(self, observation_space, hidden_size, b1_inchannels, b2_inchannels):
        super().__init__()
        
        # construct the two branches (remove the final ReLUs)
        self.branch1 = SimpleCNN(observation_space, hidden_size, in_channels=b1_inchannels).cnn
        self.branch2 = SimpleCNN(observation_space, hidden_size, in_channels=b2_inchannels).cnn
        self.branch1[-1] = nn.Identity()
        self.branch2[-1] = nn.Identity()

        self.merge = nn.Sequential(
                        nn.Linear(2*hidden_size, hidden_size),
                        nn.ELU(True))

        self.layer_init()

    def forward(self, observations):
        b1 = self.branch1(observations['rgb'])
        b2 = self.branch2(observations['aux'])
        output = self.merge(torch.cat([b1, b2], 1))
        return output

    def layer_init(self):
        for layer in [self.branch1, self.branch2, self.merge]:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)


class RGBSaliencyTwoStream(TwoStreamNetwork):
    def __init__(self, observation_space, hidden_size):
        super().__init__(observation_space, hidden_size, b1_inchannels=3, b2_inchannels=1)

class RGBAffordanceTwoStream(TwoStreamNetwork):
    def __init__(self, observation_space, hidden_size):
        super().__init__(observation_space, hidden_size, b1_inchannels=3, b2_inchannels=7)


class PolicyNetwork(Policy):
    def __init__(self, observation_space, action_space, hidden_size, **kwargs):
        super().__init__(
            BaseEncoder(
                observation_space=observation_space,
                hidden_size=hidden_size,
                visual_encoder=kwargs['vis_encoder'],
                state_encoder=RNNStateEncoder,
            ),
            action_space.n,
        )

class RandomPolicy(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size, **kwargs):
        super().__init__()
        self.num_actions = action_space.n

        # dummy network
        self.net = nn.Linear(1, 1)
        self.net.num_recurrent_layers = 1

    def load_state_dict(self, state):
        return

    def act(self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False):
        B = observations['rgb'].shape[0]
        value = torch.zeros(B, 1)
        action = torch.from_numpy(np.random.choice(self.num_actions, B)).view(-1, 1).long()
        action_log_probs = torch.zeros(B, 1)
        return value, action, action_log_probs, rnn_hidden_states
