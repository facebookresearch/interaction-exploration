# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import copy

from interaction_exploration.utils import map_util
from .thor import *

class ThorEnjoy:

    def init_environment(self):
        super().init_environment()
        self.controller.step(dict(action='ToggleMapView'))
        topdown = np.array(self.state.frame)
        cam_position = self.state.metadata["cameraPosition"]
        cam_orth_size = self.state.metadata["cameraOrthSize"]
        self.cam_params = {'cam_position':cam_position, 'cam_orth_size':cam_orth_size}
        self.controller.step(dict(action='ToggleMapView'))
        self.viz_step = {'frame':np.array(self.state.frame), 'topdown':topdown}

    def step(self, action):
        obs, rewards, done, info = super().step(action)

        # store original action success
        success = self.state.metadata['lastActionSuccess']

        # generate topdown view
        self.controller.step(dict(action='ToggleMapView'))
        topdown = np.array(self.state.frame)
        self.controller.step(dict(action='ToggleMapView'))

        # generate pose/action points
        pose = self.agent_pose(self.state)
        points = map_util.process_topdown(self.cam_params, pose)
        act_info = copy.deepcopy(self.step_info)
        act_info['pts'] = points

        # store info
        self.viz_step = {'frame':np.array(self.state.frame), 'topdown':topdown, **act_info}

        # restore original action success
        self.state.metadata['lastActionSuccess'] = success

        return obs, rewards, done, info

    def get_viz_data(self):
        return self.viz_step



class ThorEnjoyVanilla(ThorEnjoy, ThorInteractionCount):
    def __init__(self, config):
        super().__init__(config)

class ThorEnjoyCycler(ThorEnjoy, ThorInteractionCycler):
    def __init__(self, config):
        super().__init__(config)

class ThorEnjoyCyclerFixedView(ThorEnjoy, ThorInteractionCyclerFixedView):
    def __init__(self, config):
        super().__init__(config)




