# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import sys
import termios
import tty
import os
import collections
import torch
import numpy as np
from PIL import ImageDraw
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torch.nn.functional as F

from rl.common.env_utils import construct_envs, get_env_class
from rl.config import get_config
from interaction_exploration.utils import util

def get_term_character():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def add_rectangle(tensor, bbox):
    img = transforms.ToPILImage()(tensor)
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox,  outline='blue', width=3)
    tensor = transforms.ToTensor()(img)
    return tensor

class KBController(object):

    def __init__(self, config):
        self.config = config

        self.command_dict = {
            '\x1b[A': 'forward',
            'w': 'up',
            's': 'down',
            'a': 'tleft',
            'd': 'tright',
            'e': 'interact',
        }

        self.config.defrost()
        self.config.DEBUG = True
        self.config.NUM_PROCESSES = 1
        self.config.freeze()

        self.envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))
        self.observation = self.envs.reset()[0]
        
        self.act_to_idx = collections.defaultdict(lambda: -1)
        self.act_to_idx.update({act:idx for idx, act in enumerate(self.envs.call_at(0, 'get_actions'))})


        sz = 300
        N = 5
        center = ((sz//N)*(N//2), (sz//N)*(N+1)//2)
        self.center_box = [center[0], center[0], center[1], center[1]]

        colors = {'green':[7, 212, 0], 'yellow':[255, 255, 0]}
        colors = {color: torch.Tensor(colors[color]).unsqueeze(1)/255 for color in colors}
        self.colors = colors
        self.interactions = ['take', 'put', 'open', 'close', 'toggle-on', 'toggle-off', 'slice']
        self.act_to_channel = {act: idx for idx, act in enumerate(self.interactions)}


        self.render()

        print ('KB controller set up.')
        print ('↑: move forward, look: wsad, action: e')


    def next_interact_command(self):
        current_buffer = ''
        while True:
            commands = self.command_dict
            current_buffer += get_term_character()
            if current_buffer == 'q' or current_buffer == '\x03':
                break

            if current_buffer in commands:
                yield commands[current_buffer]
                current_buffer = ''
            else:
                match = False
                for k,v in commands.items():
                    if k.startswith(current_buffer):
                        match = True
                        break

                if not match:
                    current_buffer = ''

    # TODO: add action names, add more actions
    def render(self):

        event = self.envs.call_at(0, 'last_event')
        frame = torch.from_numpy(np.array(event.frame)).float().permute(2, 0, 1)/255
        frame = F.interpolate(frame.unsqueeze(0), 300, mode='bilinear', align_corners=True)[0]
        frame = add_rectangle(frame, self.center_box)

        if not self.config.SHOW_MARKERS:
            util.show_wait(frame, T=1, win='frame')
            return

        masks = self.envs.call_at(0, 'get_current_mask').byte()

        actions = ['open']
        beacon = torch.zeros(3, 300, 300)
        viz_tensors = []
        for action in actions:
            channel = self.act_to_channel[action]
            beacon.zero_()
            beacon[:, masks[channel, 0]] = self.colors['yellow']
            beacon[:, masks[channel, 1]] = self.colors['green']
            beacon = util.blend(frame, beacon)
            viz_tensors.append(beacon)

        grid = make_grid(viz_tensors, nrow=len(viz_tensors))
        util.show_wait(grid, T=1)

    

    def step(self):
        for action in self.next_interact_command():
            if action=='interact':
                prompt = ['# Options']
                prompt += ['Interactions: take, put, open, close, toggle-on, toggle-off, slice']
                prompt += ['Misc: done, reset']
                prompt += ['>> ']
                action = input('\n'.join(prompt))
            yield action

    def run(self):
        for action in self.step():

            # handle special controller actions
            if action=='done':
                sys.exit(0)

            if action=='reset':
                self.observation = self.envs.reset()
                continue

            act_idx = self.act_to_idx[action]

            if act_idx==-1:
                print ('Action not recognized')
                continue


            # handle environment actions
            outputs = self.envs.step([act_idx])
            self.observation, reward, done, info = [list(x)[0] for x in zip(*outputs)]
            print (f"A: {info['action']} | S: {info['success']} | R: {info['reward']}")


            display = os.environ['DISPLAY']
            os.environ['DISPLAY'] = os.environ['LDISPLAY']
            self.render()
            os.environ['DISPLAY'] = display

if __name__=='__main__':

    import argparse
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--env-name', default='ThorInteractionCount-v0')
    parser.add_argument('--markers', action='store_true')
    parser.add_argument('--x_display', default='0.0')
    args = parser.parse_args()


    os.environ['LDISPLAY'] = os.environ['DISPLAY']

    config = get_config()
    config.defrost()
    config.ENV_NAME = args.env_name
    config.RL.PPO.num_steps = 10000000 
    config.X_DISPLAY = args.x_display 
    config.SHOW_MARKERS = args.markers
    config.freeze()

    controller = KBController(config)
    controller.run()
