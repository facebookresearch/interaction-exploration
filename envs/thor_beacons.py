# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import collections
import numpy as np
import torch
import copy
import torch.nn.functional as F
from einops import rearrange

from interaction_exploration.utils import map_util
from .thor import ThorInteractionCount


def resize_results(frames, beacon_mask, out_sz):

    if out_sz!=frames.shape[-1]:
        frames = F.interpolate(frames, out_sz, mode='bilinear', align_corners=True)

    if out_sz!=beacon_mask.shape[-1]:
        N_interactions = beacon_mask.shape[1]
        beacon_mask = rearrange(beacon_mask, 'n i c h w -> n (i c) h w')
        beacon_mask = F.interpolate(beacon_mask, out_sz, mode='bilinear', align_corners=True)
        beacon_mask = rearrange(beacon_mask, 'n (i c) h w -> n i c h w', i=N_interactions, c=2)
        beacon_mask = beacon_mask>0

    return frames, beacon_mask

class ThorBeaconsFixedScale(ThorInteractionCount):
    def __init__(self, config):
        super().__init__(config)
        self.history = []
        self.beacons = []
        self.dist_thresh =  2e-1 # TODO: yacs
        self.mask_sz = 80 # TODO: yacs

        cam_params = (self.mask_sz//2,)*4 # fx, fy, cx, cy
        self.projector = map_util.CameraProjection(cam_params, self.mask_sz, self.rot_size_x, self.rot_size_y)

    def init_params(self):
        params = super().init_params()
        params['renderDepthImage'] = True
        return params

    def get_target_obj(self, obj_property, overlap_thresh=0.3):

        targets = super().get_target_obj(obj_property, overlap_thresh)

        if targets['int_target']=='dummy':
            targets['int_pt'] = [self.mask_sz//2, self.mask_sz//2]
            return targets

        target_obj = targets[targets['int_target']]
        seg_grid = self.state.instance_masks[target_obj['objectId']][self.center[0]:self.center[1], self.center[0]:self.center[1]] # BxB grid
        seg_grid = torch.from_numpy(seg_grid)
        nz_idx = seg_grid.nonzero(as_tuple=False) # (N, 2)
        dist, sel_idx = torch.abs((nz_idx-seg_grid.shape[0]//2)).sum(1).min(0)
        int_pixel = nz_idx[sel_idx] + self.center[0]
        targets['int_pt'] = int_pixel.tolist() # (x, y) of interaction location
        targets['int_pt'] = [int(targets['int_pt'][0]*self.mask_sz/self.frame_sz), int(targets['int_pt'][1]*self.mask_sz/self.frame_sz)]
 
        return targets

    def get_beacon_position(self, state, act_info):

        depth, P, agent_y = state['depth'], state['P'], state['agent_y']
        targets = act_info['target']

        # dummy object at the center of the screen for marking
        if targets['int_target']=='dummy':
            center_x, center_y = act_info['target']['int_pt']
            center_p = P[0, :, center_x, center_y].tolist()
            center_depth = depth[center_x, center_y]

            # place dummy marker within a certain distance
            on_floor = np.abs(center_p[1]-agent_y) < 1
            if (not on_floor and center_depth>=1.1) or (center_depth>=5 or center_depth<0.5) or (on_floor and center_depth>2):
                return None

            target_obj = {'position':{'x':center_p[0],
                                      'y':center_p[1],
                                      'z':center_p[2]},
                          'rotation':{'x':0, 'y':0, 'z':0},
                          'objectId': 'dummy', 
                         }
            act_info['target']['dummy'] = target_obj

        # calculate the marker location
        sel_x, sel_y = targets['int_pt']
        world_pt = tuple(P[0, :, sel_x, sel_y].tolist())

        target_obj = targets[targets['int_target']]
        p = (target_obj['position']['x'], -target_obj['position']['y'], target_obj['position']['z'])
        rot = (target_obj['rotation']['x'], target_obj['rotation']['y'], target_obj['rotation']['z'])

        return {'pt':world_pt, 'p':p, 'rot': rot, 'target':{'objectId':target_obj['objectId']}}


    def compile_history(self, state):
        hist = {}

        hist['frame'] = np.array(state.frame) # (300, 300, 3) [0, 255]
        hist['P'] = state.metadata['image_to_world'][0].clone() # (3, H, W)
        hist['pose'] = self.agent_pose(state)

        obj_poses = {}
        for obj in state.metadata['objects']: # don't keep all. TODO: make get obj return only visible obj list
            pos, rot = obj['position'], obj['rotation']
            obj_poses[obj['objectId']] = [pos['x'], -pos['y'], pos['z'], rot['x'], rot['y'], rot['z']]
        hist['obj_poses'] = obj_poses

        hist['inv_obj'] = self.inv_obj
        hist['visible_objs'] = set([obj['objectId'] for obj in state.metadata['objects'] if obj['objectId'] in state.instance_masks]) # TODO: just keys
        return hist


    def step(self, action):

        if 'image_to_world' not in self.state.metadata:
            self.state.metadata['image_to_world'] = self.projector.image_to_world(self.agent_pose(self.state), self.state.depth_frame)

        prev_state = {'depth': np.array(self.state.depth_frame),
                      'P': self.state.metadata['image_to_world'].clone(),
                      'agent_y': self.state.metadata['agent']['position']['y']
                      }

        obs, rewards, done, info = super().step(action)

        # place beacon for interaction
        if info['action'] in self.interaction_set:
            beacon = self.get_beacon_position(prev_state, info)
            if beacon is not None:
                # place_beacon is called AFTER the action is executed. Place for previous time-step
                self.beacons.append({'t':self.t-1, 'action': info['action'], 'success': info['success'], **beacon})

        # get image_to_world for last timestep
        self.state.metadata['image_to_world'] = self.projector.image_to_world(self.agent_pose(self.state), self.state.depth_frame)

        hist = self.compile_history(self.state)
        self.history.append({'t':self.t-1, **hist})

        # if final timestep, then compute masks for the entire trajectory and add to info
        if self.t==self.max_t:
            info['traj_masks'] = self.compute_masks()

        return obs, rewards, done, info

    # for real time viz only. Use compute_masks to collect the full dataset
    def get_current_mask(self):

        if 'image_to_world' not in self.state.metadata:
            pose = self.agent_pose(self.state)
            depth = self.state.depth_frame
            self.state.metadata['image_to_world'] = self.projector.image_to_world(pose, depth)

        hist = self.compile_history(self.state)
        history = [{'t':0, **hist}]
        frames, beacon_mask, poses, info = self.compute_masks(history, self.frame_sz)
        beacon_mask = beacon_mask[-1]
        return beacon_mask

    def update_beacon_coordinates(self, history):

        rotation_matrices = {} 
        def rot_matrix_cache(rot, inv):
            key = tuple(rot) + (inv,)
            if key in rotation_matrices:
                return rotation_matrices[key]
            return map_util.get_rotation_matrix_3D(rot, inv)

        beacon_R1_0 = torch.stack([map_util.get_rotation_matrix_3D(beacon['rot'], True) for beacon in self.beacons], 0)
        beacon_dp = torch.stack([torch.Tensor(beacon['pt']) - torch.Tensor(beacon['p']) for beacon in self.beacons], 0)

        beacons = [{'pt':beacon['pt'], 'target':{'objectId':beacon['target']['objectId']}, 'action':beacon['action'], 'success':beacon['success'], 't':beacon['t']} for beacon in self.beacons]
        # ^ throw away a lot of stuff we no longer need

        for t, hist in enumerate(history):

            beacon_t = copy.deepcopy(beacons)

            obj_poses = hist['obj_poses']
            update_idx = []
            R0_2 = []
            p2 = []
            for idx, beacon in enumerate(beacon_t):

                # fixed point on wall/floor
                if beacon['target']['objectId']=='unknown':
                    continue

                # object has changed state/disappeared
                if beacon['target']['objectId'] not in obj_poses or beacon['target']['objectId']==hist['inv_obj'] or beacon['target']['objectId'] not in hist['visible_objs']:
                    beacon['pt'] = None
                    continue

                pose = obj_poses[beacon['target']['objectId']] # (x, y, z) -- pos + (x, y, z) -- rot
                R0_2.append(rot_matrix_cache(pose[3:], False))
                p2.append(pose[:3])
                update_idx.append(idx)


            if len(update_idx)==0:
                hist['beacons'] = [beacon for beacon in beacon_t if beacon['pt'] is not None]
                continue

            R1_0 = beacon_R1_0[update_idx]
            dp = beacon_dp[update_idx].unsqueeze(-1)


            R0_2 = torch.stack(R0_2, 0)
            R1_2 = torch.bmm(R1_0, R0_2)
            p2 = torch.Tensor(p2)

            dp = torch.bmm(R1_2, dp)[:, :, 0] # local (x, y, z)
            new_p = dp + p2
            
            for enum_idx, t in enumerate(update_idx):
                beacon_t[t]['pt'] = tuple(new_p[enum_idx].tolist())

            hist['beacons'] =[beacon for beacon in beacon_t if beacon['pt'] is not None]

        return history

    def compute_masks(self, history=None, out_sz=None):

        info = {'scene':self.scene, 'episode':self.episode_id}
        out_sz = out_sz or self.mask_sz
        history = history or self.history

        if len(history)==0 or len(self.beacons)==0:
            return torch.zeros(1, 3, out_sz, out_sz), torch.zeros(1, len(self.interactions), 2, out_sz, out_sz).byte(), torch.zeros(1, 5), info

        # each hist in history will have different beacons
        history = self.update_beacon_coordinates(history) # 4.2s

        P = torch.stack([hist['P'] for hist in history], 0) # (T=128, 3, H, W)) --> 511 x 300*300*3

        # get wdist to every beacon (across time), and maintain a pt-->index map
        uniq_beacon_pts = set()
        for t, hist in enumerate(history):
            uniq_beacon_pts |= set([beacon['pt'] for beacon in hist['beacons']])
        uniq_beacon_pts = [(0, 0, 0)] if len(uniq_beacon_pts)==0 else sorted(uniq_beacon_pts)
        wdist_pt_to_idx = {pt:idx for idx, pt in enumerate(uniq_beacon_pts)}

        B = torch.Tensor(uniq_beacon_pts) # (N, 3)
        P_flat = rearrange(P, 't p h w -> (t h w) p') # (THW, 3) torch.Size([45990000, 3])

        wdist = torch.cdist(P_flat, B) # (THW, N)
        wdist = rearrange(wdist, '(t h w) n -> t h w n', t=P.shape[0], h=self.mask_sz, w=self.mask_sz) # (T, H, W, N)

        pose_last_seen_time = {hist['pose']:-1 for hist in history}
        beacon_history = []
        for t, hist in enumerate(history):

            beacons_t = hist['beacons']
            if len(beacons_t)==0:
                continue

            # Seen this pose before but state has not changed
            if pose_last_seen_time[hist['pose']] != -1:
                positive_beacons_since_last_visit = [beacon for beacon in beacons_t if beacon['t']>pose_last_seen_time[hist['pose']] and beacon['t']<=t and beacon['success']]
                if len(positive_beacons_since_last_visit)==0:
                    continue

            pose_last_seen_time[hist['pose']] = t
            hist['wdist'] = wdist[t]
            beacon_history.append(hist)

        if len(beacon_history)==0:
            return torch.zeros(1, 3, out_sz, out_sz), torch.zeros(1, len(self.interactions), 2, out_sz, out_sz).byte(), torch.zeros(1, 5), info


        beacon_mask = torch.zeros(len(beacon_history), len(self.interactions), 2, self.mask_sz, self.mask_sz)

        for ch, action in enumerate(self.interactions):

            for t in range(len(beacon_history)):

                hist = beacon_history[t]
                pos_inds_t = [wdist_pt_to_idx[beacon['pt']] for beacon in hist['beacons'] if beacon['action']==action and beacon['success']]
                neg_inds_t = [wdist_pt_to_idx[beacon['pt']] for beacon in hist['beacons'] if beacon['action']==action and not beacon['success']]

                if len(neg_inds_t)>0:
                    neg_mask = hist['wdist'][:, :, neg_inds_t].min(2)[0] < self.dist_thresh  # (H, W)
                    beacon_mask[t, ch, 0][neg_mask] = 1

                if len(pos_inds_t)>0:
                    pos_mask = hist['wdist'][:, :, pos_inds_t].min(2)[0] < self.dist_thresh  # (H, W)
                    beacon_mask[t, ch, 1][pos_mask] = 1

        #-----------------------------------------------------------------------------------------------------------------#

        frames = np.stack([hist['frame'] for hist in beacon_history], 0)
        frames = torch.from_numpy(frames).float().permute(0, 3, 1, 2)/255
        poses = torch.Tensor([hist['pose'] for hist in beacon_history])
        frames, beacon_mask = resize_results(frames, beacon_mask, out_sz)
        
        return frames, beacon_mask, poses, info


    def reset(self):
        obs = super().reset()
        self.beacons = []
        self.history = []
        return obs

class ThorBeaconsObjects(ThorInteractionCount):
    def __init__(self, config):
        super().__init__(config)
        self.history = []
        self.beacons = []
        self.mask_sz = 80

    def compile_history(self, state):
        hist = {}
        hist['instance_masks'] =  dict(state.instance_masks)
        hist['inv_obj'] = state.metadata['inventoryObjects'][0] if len(state.metadata['inventoryObjects'])>0 else None
        hist['frame'] = np.array(state.frame)
        hist['pose'] = self.agent_pose(state)
        hist['visible_objs'] = set([obj['objectId'] for obj in state.metadata['objects'] if obj['objectId'] in state.instance_masks])
        return hist

    def step(self, action):

        obs, rewards, done, info = super().step(action)

        # place beacon for interaction
        if info['action'] in self.interaction_set and info['target'] is not None:
            target = info['target']['objectId'] or info['target']['center_objectId']
            if target is not None:
                self.beacons.append({'t':self.t-1, 'target':target, 'action':info['action'], 'success':info['success']})

        hist = self.compile_history(self.state)
        self.history.append({'t':self.t-1, **hist})

        # if final timestep, then compute masks for the entire trajectory and add to info
        if self.t==self.max_t:
            info['traj_masks'] = self.compute_masks()

        return obs, rewards, done, info

    # for real time viz only. Use compute_masks to collect the full dataset
    def get_current_mask(self):
        hist = self.compile_history(self.state)
        history = [{'t':0, **hist}]
        frames, beacon_mask, poses, info = self.compute_masks(history, 300)
        beacon_mask = beacon_mask[-1]
        return beacon_mask


    def compute_masks(self, history=None, out_sz=None):

        info = {'scene':self.scene, 'episode':self.episode_id}
        out_sz = out_sz or self.mask_sz
        history = history or self.history

        if len(history)==0:
            return torch.zeros(1, 3, out_sz, out_sz), torch.zeros(1, len(self.interactions), 2, out_sz, out_sz).byte(), torch.zeros(1, 5), info

        # Discard timesteps with redundant positions (same visual content)
        # each time-step will admit only some beacons
        positive_beacons = [beacon for beacon in self.beacons if beacon['success']]

        pose_last_seen_time = {hist['pose']:-1 for hist in history}
        beacon_history = []
        for t, hist in enumerate(history):

            # Seen this pose before but state has not changed
            if pose_last_seen_time[hist['pose']] != -1:
                positive_beacons_since_last_visit = [beacon for beacon in positive_beacons if beacon['t']>pose_last_seen_time[hist['pose']] and beacon['t']<=t and beacon['target'] in hist['visible_objs']]
                if len(positive_beacons_since_last_visit)==0:
                    continue

            pose_last_seen_time[hist['pose']] = t
            beacon_history.append(hist)

        #-----------------------------------------------------------------------------------------------------------------#

        beacon_mask = torch.zeros(len(beacon_history), len(self.interactions), 2, self.frame_sz, self.frame_sz)

        beacons_by_key = collections.defaultdict(list)
        for beacon in self.beacons:
            key = (beacon['action'], beacon['success'])
            beacons_by_key[key].append(beacon)

        for t, hist in enumerate(beacon_history):

            inv_obj = hist['inv_obj']
            for ch, action in enumerate(self.interactions):
                pos_masks = [hist['instance_masks'][beacon['target']] for beacon in beacons_by_key[(action, True)] if beacon['target'] in hist['instance_masks'] and beacon['target']!=inv_obj]
                neg_masks = [hist['instance_masks'][beacon['target']] for beacon in beacons_by_key[(action, False)] if beacon['target'] in hist['instance_masks'] and beacon['target']!=inv_obj]

                if len(neg_masks)>0:
                    neg_mask = torch.from_numpy(sum(neg_masks).astype(bool)).byte()
                    beacon_mask[t, ch, 0][neg_mask] = 1
                if len(pos_masks)>0:
                    pos_mask = torch.from_numpy(sum(pos_masks).astype(bool)).byte()
                    beacon_mask[t, ch, 1][pos_mask] = 1


        frames = np.stack([hist['frame'] for hist in beacon_history], 0)
        frames = torch.from_numpy(frames).float().permute(0, 3, 1, 2)/255
        poses = torch.Tensor([hist['pose'] for hist in beacon_history])
        frames, beacon_mask = resize_results(frames, beacon_mask, out_sz)
            
        return frames, beacon_mask, poses, info


    def reset(self):
        obs = super().reset()
        self.beacons = []
        self.history = []
        return obs


