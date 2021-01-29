# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
import copy
import networkx as nx
import numpy as np
import itertools
import torch
import collections
import tqdm
import os

from envs.thor import ThorInteractionCount
from ..config import get_config

# state = each action (open, close, put, take) per object
class ThorNX(ThorInteractionCount):

    def __init__(self, config):
        super().__init__(config)
        self.queue = []

    def get_path_to_obj(self, target_obj):
        
        env_y = list(self.G.nodes())[0][1]

        event = self.controller.step(dict(action='PositionsFromWhichItemIsInteractable', objectId=target_obj['objectId']))
        int_poses = event.metadata['actionReturn']
        int_poses = zip(int_poses['x'], int_poses['y'], int_poses['z'], int_poses['rotation'], int_poses['horizon'])
        # int_poses = [(x, y, z, self.rot_size_y*np.round(rot/self.rot_size_y), self.rot_size_x*np.round(hor/self.rot_size_x)) for x, y, z, rot, hor in int_poses]
    
        parsed_poses = []
        for x, y, z, rot, hor in int_poses:
            rot = self.rot_size_y*np.round(rot/self.rot_size_y)
            hor = self.rot_size_x*np.round(hor/self.rot_size_x)
            if hor == 330:
                hor = -30
            parsed_poses.append((x, env_y, z, rot, hor))


        int_poses = [pose for pose in parsed_poses if pose in self.G]

        if len(int_poses)==0:
            return None

        target = min(int_poses, key=lambda pos: np.sqrt((pos[0]-target_obj['position']['x'])**2 + (pos[2]-target_obj['position']['z'])**2))

        pose = list(self.agent_pose(self.state))
        pose[1] = env_y
        source = tuple(pose)

        path = nx.shortest_path(self.G, source=source, target=target, weight=None, method='dijkstra')
        
        return path


    def compute_graph(self):

        G = nx.DiGraph()

        queue = [self.agent_pose(self.state)] # random reachable pose
        seen = set(queue)
        while True:

            if len(queue)==0:
                break

            node = queue.pop(0)
            x0, y0, z0, rot0, hor0 = node

            nbhs = []

            # turn
            nbhs.append({'act':'tright', 'pos':node[:3]+((rot0+30)%360, hor0)})      
            nbhs.append({'act':'tleft', 'pos':node[:3]+((rot0-30)%360, hor0)})            

            # look
            nbhs.append({'act':'up', 'pos':node[:3]+(rot0, max(-30, hor0-15))})      
            nbhs.append({'act':'down', 'pos':node[:3]+(rot0, min(60, hor0+15))})    

            # move -- only allow if facing right angles
            if rot0%90 == 0:
                x = x0 + 0.25*np.sin(np.radians(rot0))
                z = z0 + 0.25*np.cos(np.radians(-rot0))
                y = y0
                x_snap = min(self.reachable_x, key=lambda item: np.abs(item-x))
                z_snap = min(self.reachable_z, key=lambda item: np.abs(item-z))

                dist = np.sqrt((x-x_snap)**2 + (z-z_snap)**2)
                if (x_snap, y, z_snap) in self.reachable_positions and dist<1e-12:
                    nbhs.append({'act':'forward', 'pos':(x_snap, y, z_snap, rot0, hor0)})    


            for nbh in nbhs:
                if nbh['pos'] not in seen:
                    queue.append(nbh['pos'])

                seen.add(nbh['pos'])
                G.add_edge(node, nbh['pos'], action=nbh['act'])

        print (f'Graph constructed with {len(G.nodes)} nodes |{len(self.reachable_positions)} reachable positions')

        # checking if all positions are in graph
        success = True
        for (x, y, z), rot, hor in itertools.product(self.reachable_positions, [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330], [-30, -15, 0, 15, 30, 45, 60]):
            if (x, y, z, rot, hor) not in G:
                print (x, y, z, rot, hor, '<-- MISSING')
                success = False
                break

        if not nx.is_strongly_connected(G):
            cc = nx.strongly_connected_components(G)
            print ('NOT STRONGLY CONNECTED:', len(cc), 'components')
            return None
        else:
            print ('IS CONNECTED!!')


        if not success:
            G = None
        return G

    def init_environment(self):

        scene, episode = self.config.test_episode
        print (f'INIT: {scene} episode {episode}')

        self.init_scene_and_agent(scene=scene, episode=episode)

        # store all reachable x, z for snapping to grid
        self.reachable_x = list(set([pos[0] for pos in self.reachable_positions]))
        self.reachable_z = list(set([pos[2] for pos in self.reachable_positions]))

        # get a list of every possible interaction that can be done (by object)
        # go to the closest object and cycle through these
        # actions = ['open', 'close', 'toggle-on', 'toggle-off', 'take', 'put', 'slice']
        interaction_candidates = []
        for obj in self.state.metadata['objects']:

            ints = []
            if obj['openable']:
                ints += ['open', 'close']
            if obj['toggleable']:
                ints += ['toggle-on', 'toggle-off']
            if obj['pickupable']:
                ints += ['take', 'put']
            if obj['sliceable']:
                ints += ['slice']

            obj = copy.deepcopy(obj)
            obj['ints'] = ints
            obj['objId'] = obj['objectId']

            if len(ints)>0:
                interaction_candidates.append(obj)
        self.interaction_candidates = interaction_candidates

    def select_next_obj(self):


        # keep objects that are still there
        obj_to_dist = {obj['objectId']:obj['distance'] for obj in self.state.metadata['objects']}

        self.interaction_candidates = [obj for obj in self.interaction_candidates if obj['objectId'] in obj_to_dist]

        if len(self.interaction_candidates)==0:
            return None

        # find nearest obj in the list        
        self.interaction_candidates = sorted(self.interaction_candidates, key = lambda obj: obj_to_dist[obj['objectId']])
        target_obj = self.interaction_candidates.pop(0)
        # print (f'Selected {target_obj["objectId"]}')
        # print (f'Possible interactions = {target_obj["ints"]}')
        return target_obj


    # [{'reward': 0, 'action': 'down', 'target': None, 'success': True}] xT
    def run(self, G, T=1024):

        self.G = G

        step_infos = []
        while len(self.interaction_candidates)>0 and self.t<T:

            path = None
            while path is None:
                target_obj = self.select_next_obj()
                if target_obj is None:
                    break
                path = self.get_path_to_obj(target_obj)

            if path is None:
                print ('DONE')
                break

            self.t += len(path) - 1
            step_infos += [{'action':'move', 'target':None, 'success':True, 'reward':0}]*(len(path) - 1)

            if self.t >= T:
                print ('Timeout')
                break

            for interaction in target_obj['ints']:

                if interaction != 'put':
                    self.step_info = {'action':interaction, 'target':{'objectId': target_obj['objectId']}, 'success':True}
                elif interaction == 'put':
                    if target_obj['parentReceptacles'] is None:
                        continue
                    self.step_info = {'action':interaction, 'target':{'objectId': target_obj['parentReceptacles'][0]}, 'success':True}

                self.step_info['reward'] = self.get_reward(None)
                step_infos.append(self.step_info)
                self.t += 1

                if self.t >= T:
                    print ('Timeout')
                    break  

            end_pos = path[-1]
            self.controller.step(dict(action='TeleportFull', x=end_pos[0], y=end_pos[1], z=end_pos[2], rotation=end_pos[3], horizon=end_pos[4]))

        # padding
        step_infos = step_infos[:T]
        step_infos += [{'action':None, 'target':None, 'success':None, 'reward':0}]*(T - len(step_infos))

        return step_infos


    def reset(self):
        obs = super().reset()
        self.queue = []
        return obs


def generate_navgraphs(env, out_fl):

    scenes = [f'FloorPlan{idx}' for idx in range(1, 31)]
    graphs = {}
    for scene in scenes:
        retry = 0
        while True:
            print (scene, retry)
            env.config.test_episode = (scene, np.random.randint(10000))
            env.reset()
            G = env.compute_graph()
            if G is not None:
                graphs[scene] = G
                break
            retry += 1

    # torch.save(graphs, out_fl)
    return graphs

if __name__=='__main__':
    config = get_config(None, None)
    config.defrost()
    config.MODE = 'eval'
    config.X_DISPLAY = '0'
    env = ThorNX(config)

    os.makedirs('interaction_exploration/cv/oracle/run0/eval/', exist_ok=True)
    navgraph_fl = 'interaction_exploration/cv/oracle/navgraphs.pth'
    if not os.path.exists(navgraph_fl):
        generate_navgraphs(env, navgraph_fl)

    test_episodes = json.load(open('interaction_exploration/data/test_episodes_K_16.json'))
    navgraphs = torch.load(navgraph_fl)

    test_episode_stats = []
    for scene, episode in tqdm.tqdm(test_episodes, total=len(test_episodes)):
        env.config.test_episode = (scene, episode)
        env.reset()
        stats = {'step_info': env.run(navgraphs[scene])} 
        episode_stats = {'scene_id':scene, 'episode_id':episode, 'stats':stats}
        test_episode_stats.append(episode_stats)
        # torch.save(episode_stats, f'interaction_exploration/cv/oracle/run0/eval/{scene}_{episode}.pth')
    # test_episode_stats = [torch.load(fl) for fl in glob.glob('interaction_exploration/cv/oracle/run0/eval/*.pth')]

    interactions_per_scene = collections.defaultdict(list)
    for stats in test_episode_stats:
        rewards = sum([step['reward'] for step in stats['stats']['step_info']])
        interactions_per_scene[stats['scene_id']].append(rewards)

    for scene in interactions_per_scene:
        interactions_per_scene[scene] = int(np.mean(interactions_per_scene[scene]))

    print (interactions_per_scene)




