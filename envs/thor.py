# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import gym
from gym import spaces
import collections
import numpy as np
import numbers
from PIL import Image
import itertools
import ai2thor.controller

class ThorEnv(gym.Env):

    def __init__(self, config):
        self.config = config
        self.x_display = config.X_DISPLAY

        self.obs_sz = self.config.ENV.OBS_SZ # 80
        self.rot_size_x = self.config.ENV.ROT_SIZE_X # 15
        self.rot_size_y = self.config.ENV.ROT_SIZE_Y # 30
        self.frame_sz = self.config.ENV.FRAME_SIZE # 300 (fixed)
        self.max_t = self.config.ENV.NUM_STEPS

        self.actions, self.action_fns = self.get_action_fns()
        self.act_to_idx = {act: idx for idx, act in enumerate(self.actions)}

        self.observation_space = spaces.Dict({'rgb': spaces.Box(-np.inf, np.inf, (3, self.obs_sz, self.obs_sz))})
        self.action_space = spaces.Discrete(len(self.actions))

        local_exe = None if self.config.ENV.LOCAL_EXE=='None' else self.config.ENV.LOCAL_EXE
        self.controller = ai2thor.controller.Controller(quality='Ultra',
                                                        local_executable_path=local_exe,
                                                        x_display=self.x_display)

    def seed(self, seed):
        self.rs = np.random.RandomState(seed)

    def get_action_fns(self):
        action_fns = {
            'forward':self.move,
            'up':self.look,
            'down':self.look,
            'tright':self.turn,
            'tleft':self.turn,
        }
        actions = ['forward', 'up', 'down', 'tright', 'tleft']
        return actions, action_fns

    def init_params(self):
        params = {
            'gridSize': 0.25,
            'renderObjectImage': False,
            'renderDepthImage': False,
        }
        return params

    def parse_action(self, action):

        if isinstance(action, np.ndarray):
            action = action.item()
        if isinstance(action, numbers.Number):
            action = self.actions[action]
        assert isinstance(action, str), 'action must be a string'

        return action

    def get_observation(self, state):
        img = state.frame
        return {'rgb': img}

    def agent_pose(self, state):
        agent = state.metadata['agent']
        pose = (agent['position']['x'], agent['position']['y'], agent['position']['z'],
               self.rot_size_y*np.round(agent['rotation']['y']/self.rot_size_y),
               self.rot_size_x*np.round(agent['cameraHorizon']/self.rot_size_x))
        return pose

    @property
    def state(self):
        return self.controller.last_event

    def move(self, direction):
        act_params = dict(action='MoveAhead')
        return {'params': act_params}

    def turn(self, direction):
        rotation = self.state.metadata['agent']['rotation']['y']
        horizon = self.state.metadata['agent']['cameraHorizon']
        if direction=='tright':
            rotation += self.rot_size_y
        elif direction=='tleft':
            rotation -= self.rot_size_y

        act_params = dict(action='RotateLook', rotation=rotation, horizon=horizon)

        return {'params': act_params}

    def look(self, direction):
        rotation = self.state.metadata['agent']['rotation']['y']
        horizon = self.state.metadata['agent']['cameraHorizon']
        if direction=='up':
            horizon -= self.rot_size_x
        elif direction=='down':
            horizon += self.rot_size_x

        act_params = dict(action='RotateLook', rotation=rotation, horizon=horizon)

        return {'params': act_params}


    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.t>=self.max_t

    def act(self, action):

        # get action parameters
        action_info = {'action': action, 'success':False}
        action_info.update(self.action_fns[action](action))

        if action_info['params'] is not None:
            self.controller.step(action_info['params'])
            action_info['success'] = self.state.metadata['lastActionSuccess']

        # if it's a movement action, double check that you're still on the grid
        if action_info['action']=='forward' and action_info['success']:
            x, y, z, rot, hor = self.agent_pose(self.state)
            if (x, y, z) not in self.reachable_positions:
                gpos = min(self.reachable_positions, key=lambda p: (p[0]-x)**2 + (p[2]-z)**2)
                self.controller.step(dict(action='TeleportFull', x=gpos[0], y=gpos[1], z=gpos[2], rotation=rot, horizon=hor))

        return action_info


    def step(self, action, **kwargs):
        self.t += 1

        action = self.parse_action(action['action'])
        self.step_info = self.act(action)

        observations = self.get_observation(self.state)
        reward = self.get_reward(observations)
        done = self.get_done(observations)

        self.step_info.update({'reward':reward, 'done':done})

        return observations, reward, done, self.step_info

    def reset(self):
        self.t = 0
        self.init_environment()
        return self.get_observation(self.state)

    def init_scene_and_agent_debug(self):
        self.scene = 'FloorPlan1'
        self.controller.reset(self.scene)
        self.controller.step(dict(action='Initialize', **self.init_params()))
        self.controller.step(dict(action='GetReachablePositions'))
        self.reachable_positions = set([(pos['x'], pos['y'], pos['z']) for pos in self.state.metadata['reachablePositions']])

        self.controller.step(dict(action='TeleportFull', x=0, y=0.9009991, z=2.25, rotation=180, horizon=0))
        self.episode_id = 0
        return

    def randomize_objects(self):
        self.controller.step(dict(action='InitialRandomSpawn',
                        randomSeed=self.episode_id,
                        forceVisible=False,
                        numPlacementAttempts=5))
                    

    def init_scene_and_agent(self, scene, episode=None):

        if self.config.DEBUG == True:
            self.init_scene_and_agent_debug()
            return 

        self.scene = scene
        self.episode_id = episode or self.rs.randint(1000000000)
        
        self.controller.reset(self.scene) 
        self.controller.step(dict(action='Initialize', **self.init_params()))

        self.randomize_objects()

        self.controller.step(dict(action='GetReachablePositions'))
        reachable_positions = [(pos['x'], pos['y'], pos['z']) for pos in self.state.metadata['reachablePositions']]
        
        init_rs = np.random.RandomState(self.episode_id)
        rot = init_rs.choice([i*self.rot_size_y for i in range(360//self.rot_size_y)])
        pos = reachable_positions[init_rs.randint(len(reachable_positions))]
        self.controller.step(dict(action='TeleportFull', x=pos[0], y=pos[1], z=pos[2], rotation=rot, horizon=0))
        
        self.reachable_positions = set(reachable_positions)

    def init_environment(self):

        scene, episode = None, None
        
        if self.config.MODE == 'train':
            scene = self.rs.choice(['FloorPlan%d'%idx for idx in range(6, 30+1)]) # 6 --> 30 = train. 1 --> 5 = test

        elif self.config.MODE == 'eval':
            if not hasattr(self, 'test_episodes'):
                self.test_episodes = iter(self.config.ENV.TEST_EPISODES)
            scene, episode = next(self.test_episodes)
            print ('INIT: %s episode %s'%(scene, episode))
        
        self.init_scene_and_agent(scene=scene, episode=episode)

        # store all reachable x, z for snapping to grid
        self.reachable_x = list(set([pos[0] for pos in self.reachable_positions]))
        self.reachable_z = list(set([pos[2] for pos in self.reachable_positions]))

        # assert self.state.frame.sum()==0, "ERR: Image frames are being rendered incorrectly"  

    # to keep track of fixed test set stats
    @property
    def current_episode(self):
        assert self.scene is not None, "ERR: Env must be initialized first"
        return {'scene_id': self.scene, 'episode_id': self.episode_id}

    # custom functions to get some data for kb_agent
    def last_event(self):
        return self.controller.last_event

    def get_actions(self):
        return self.actions

    def render(self, mode='human'):
        pass

    def close(self):
        self.controller.stop()

class ThorObjs(ThorEnv):

    def __init__(self, config):
        super().__init__(config)
        self.movement_actions = ['forward', 'up', 'down', 'tright', 'tleft']
        self.interactions = ['take', 'put', 'open', 'close', 'toggle-on', 'toggle-off', 'slice']
        self.interaction_set = set(self.interactions)

        self.N = self.config.ENV.NGRID # 5x5 grid, center = active
        self.center = ((self.frame_sz//self.N)*(self.N//2), (self.frame_sz//self.N)*(self.N+1)//2)
        self.center_grid = np.array([[self.center[0], self.center[0], self.center[1], self.center[1]]]) # xyxy 

    def get_action_fns(self):
        actions, action_fns = super().get_action_fns()
        action_fns.update({
            'take': self.take,
            'put': self.put,
            'open': self.open_obj,
            'close': self.close_obj,
            'toggle-on': self.toggle_on,
            'toggle-off': self.toggle_off,
            'slice': self.slice,
        })
        actions += ['take', 'put', 'open', 'close', 'toggle-on', 'toggle-off', 'slice']
        return actions, action_fns

    def init_params(self):
        params = super().init_params()
        params['renderObjectImage'] = True
        params['visibilityDistance'] = 1.0
        return params

    @property
    def inv_obj(self):
        inventory = self.state.metadata['inventoryObjects']
        return inventory[0]['objectId'] if len(inventory) > 0 else None

    # get the object at the center of the screen + object that satisfies the action property
    def get_target_obj(self, obj_property, overlap_thresh=0.3):

        objId_to_obj = {obj['objectId']:obj for obj in self.state.metadata['objects'] if obj['visible'] and obj['objectId']!=self.inv_obj}

        instance_segs = self.state.instance_segmentation_frame # (300, 300, 3)
        color_to_count = Image.fromarray(instance_segs, 'RGB').getcolors()
        color_to_count = dict({pix:cnt for cnt,pix in color_to_count})
        color_to_objId = self.state.color_to_object_id

        active_px = instance_segs[self.center[0]:self.center[1], self.center[0]:self.center[1]] # (B, B, 3)
        S = active_px.shape[0]
        instance_counter = collections.defaultdict(list)
        for i, j in itertools.product(range(S), range(S)):
            color = tuple(active_px[i, j])    
            if color not in color_to_objId or color_to_objId[color] not in objId_to_obj:
                continue            
            instance_counter[color].append(np.abs(i-S//2) + np.abs(j-S//2))
        instance_counter = [{'color':color, 'N':len(scores), 'objectId':color_to_objId[color], 'dist':np.mean(scores), 'p1':len(scores)/S**2, 'p2':len(scores)/color_to_count[color]} for color, scores in instance_counter.items()]

        # either >K% of the object is inside the box, OR K% of the pixels belong to that object
        all_targets = [inst for inst in instance_counter if inst['p1']>overlap_thresh or inst['p2']>overlap_thresh]
        all_targets = sorted(all_targets, key=lambda x: x['dist'])
        act_targets = [candidate for candidate in all_targets if obj_property(objId_to_obj[candidate['objectId']])]
        
        targets = {'objectId':None, 'obj':None, 'center_objectId':None, 'center_obj':None, 'int_target':None}
        if len(all_targets)>0:
            objId = all_targets[0]['objectId']
            targets.update({'center_objectId':objId, 'center_obj':objId_to_obj[objId], 'int_target':'center_obj'})

        if len(act_targets)>0:
            objId = act_targets[0]['objectId']
            targets.update({'objectId':objId, 'obj':objId_to_obj[objId], 'int_target':'obj'})

        if targets['int_target'] is None:
            targets['int_target'] = 'dummy'

        return targets

    def take(self, action):

        obj_property = lambda obj: obj['pickupable']
        target_obj = self.get_target_obj(obj_property)

        act_params = None
        if target_obj['objectId'] is not None:
            act_params = dict(action='PickupObject', objectId=target_obj['objectId'])

        act_info = {'target':target_obj, 'params':act_params}

        return act_info

    def put(self, action):

        obj_property = lambda obj: obj['receptacle'] and (obj['openable'] and obj['isOpen'] or not obj['openable'])
        target_obj = self.get_target_obj(obj_property, overlap_thresh=0.1) # easier to put things down

        act_params = None
        if target_obj['objectId'] is not None:
            act_params = dict(action='PutObject', forceAction=True, objectId=self.inv_obj, receptacleObjectId=target_obj['objectId'])

        act_info = {'held_obj':self.inv_obj, 'target':target_obj, 'params':act_params}

        return act_info

    def open_obj(self, action):

        obj_property = lambda obj: obj['openable'] and not obj['isOpen']
        target_obj = self.get_target_obj(obj_property)

        act_params = None
        if target_obj['objectId'] is not None:
            act_params = dict(action='OpenObject', objectId=target_obj['objectId'])
            
        act_info = {'target':target_obj, 'params':act_params}

        return act_info

    def close_obj(self, action):

        obj_property = lambda obj: obj['openable'] and obj['isOpen']
        target_obj = self.get_target_obj(obj_property)

        act_params = None
        if target_obj['objectId'] is not None:
            act_params = dict(action='CloseObject', objectId=target_obj['objectId'])
            
        act_info = {'target':target_obj, 'params':act_params}

        return act_info

    def toggle_on(self, action):

        obj_property = lambda obj: obj['toggleable'] and not obj['isToggled']
        target_obj = self.get_target_obj(obj_property)

        act_params = None
        if target_obj['objectId'] is not None:
            act_params = dict(action='ToggleObjectOn', objectId=target_obj['objectId'])
            
        act_info = {'target':target_obj, 'params':act_params}

        return act_info

    def toggle_off(self, action):

        obj_property = lambda obj: obj['toggleable'] and obj['isToggled']
        target_obj = self.get_target_obj(obj_property)

        act_params = None
        if target_obj['objectId'] is not None:
            act_params = dict(action='ToggleObjectOff', objectId=target_obj['objectId'])
            
        act_info = {'target':target_obj, 'params':act_params}

        return act_info

    def slice(self, action):

        obj_property = lambda obj: obj['sliceable']
        target_obj = self.get_target_obj(obj_property)

        # can Slice only when holding a knife
        inventory_objects = self.state.metadata['inventoryObjects']
        holding_knife = len(inventory_objects)>0 and 'Knife' in inventory_objects[0]['objectType']

        act_params = None
        if target_obj['objectId'] is not None and holding_knife:
            act_params = dict(action='SliceObject', objectId=target_obj['objectId'])
            
        act_info = {'target':target_obj, 'params':act_params}

        return act_info



# state = each action (open, close, put, take) per object
class ThorInteractionCount(ThorObjs):

    def __init__(self, config):
        super().__init__(config)
        self.state_count = collections.defaultdict(int)

    def get_reward(self, observations):
        reward = 0

        info = self.step_info
        if info['action'] in self.interaction_set and info['target'] is not None and info['success']:
            key = (info['action'], info['target']['objectId'])
            if key not in self.state_count:
                reward += 1.0
                self.state_count[key] += 1

        return reward

    def reset(self):
        obs = super().reset()
        self.state_count = collections.defaultdict(int)
        return obs


class ThorInteractionCycler(ThorInteractionCount):
    def __init__(self, config):
        super().__init__(config)
        self.interaction_queue = []
        self.phase = 'move'
        self.nav_iter = 0
        self.max_nav_iter = 5

    def step(self, action, **kwargs):

        if self.phase=='move':
              
            self.nav_iter += 1
            if self.nav_iter == self.max_nav_iter:
                self.phase = 'interact'
                self.interaction_queue = list(self.interactions)

        elif self.phase=='interact':

            act_str = self.interaction_queue.pop(0)
            action['action'] = self.actions.index(act_str)
                        
            if len(self.interaction_queue)==0:
                self.phase = 'move'
                self.nav_iter = 0

        return super().step(action, **kwargs)

    def reset(self):
        obs = super().reset()
        self.interaction_queue = []
        self.phase = 'move'    
        self.nav_iter = 0
        return obs

class ThorInteractionCyclerFixedView(ThorInteractionCycler):
    def __init__(self, config):
        super().__init__(config)

    def look(self, direction):
        return {'params': None}

    def reset(self):
        super().reset()
        _, _, _, rot, hor = self.agent_pose(self.state) 
        self.controller.step(dict(action='RotateLook', rotation=rot, horizon=30))
        obs = self.get_observation(self.state)
        return obs


class ThorNavigationNovelty(ThorObjs):

    def __init__(self, config):
        super().__init__(config)
        self.state_count = collections.defaultdict(int)

    def get_reward(self, observations):
        pose = self.agent_pose(self.state)
        # key = pose
        key = (pose[0], pose[2]) # (x, z)

        reward = 0.1/np.sqrt(self.state_count[key]+1)
        self.state_count[key] += 1
        return reward

    def reset(self):
        obs = super().reset()
        self.state_count = collections.defaultdict(int)
        return obs

class ThorObjectCoverage(ThorObjs):

    def __init__(self, config):
        super().__init__(config)
        self.seen = collections.defaultdict(int)

    # Find target object in center of the screen
    def get_reward(self, observations):
        reward = 0    
        target = self.get_target_obj(lambda obj: False) 
        if target['int_target']=='center_obj' and self.seen[target['center_objectId']] == 0:
            self.seen[target['center_objectId']] += 1
            reward += 1
        return reward

    def reset(self):
        obs = super().reset()
        self.seen = collections.defaultdict(int)
        return obs

