# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import math
import copy
from einops import rearrange

def get_rotation_matrix(axis, angle_deg):

    angle_rad  = 2 * math.pi * (angle_deg / 360.0)

    c = math.cos(angle_rad)
    s = math.sin(angle_rad)

    if axis=='x':
        R = torch.Tensor(
            [[1, 0, 0],
             [0, c, -s],
             [0, s, c]])

    elif axis=='y':
        R = torch.Tensor(
                [[c, 0, s],
                 [0, 1, 0],
                 [-s, 0, c]])


    elif axis=='z':
        R = torch.Tensor(
            [[c, -s, 0],
             [s, c, 0],
             [0, 0, 1]])


    return R.unsqueeze(0) # (1, 3, 3)


def get_rotation_matrix_3D(angles, inv=False):

    angles = [2 * math.pi * (angle / 360.0) for angle in angles]
    cx, sx = math.cos(angles[0]), math.sin(angles[0])
    cy, sy = math.cos(angles[1]), math.sin(angles[1])
    cz, sz = math.cos(angles[2]), math.sin(angles[2])

    R = torch.Tensor(
            [[cy*cz+sx*sy*sz ,  cz*sx*sy-cy*sz, cx*sy],
             [cx*sz          ,  cx*cz         , -sx  ],
             [-cz*sy+cy*sx*sz,  cy*cz*sx+sy*sz, cx*cy],
            ])

    if inv:
        R = R.inverse()

    return R


class CameraProjection:

    def __init__(self, cam_params, out_size, rot_size_x, rot_size_y):
        self.cam_params = cam_params # fx, fy, cx, cy
        self.out_size = out_size
        self.rot_x = torch.cat([get_rotation_matrix('x', theta) for theta in range(0, 360, rot_size_x)], 0) # (N, 3, 3)
        self.rot_y = torch.cat([get_rotation_matrix('y', theta) for theta in range(0, 360, rot_size_y)], 0) # (N, 3, 3)
        
        self.rot_size_x = rot_size_x
        self.rot_size_y = rot_size_y

    def image_to_world(self, agent_pose, depth_inputs):

        agent_pose = torch.Tensor(agent_pose)
        depth_inputs = torch.from_numpy(depth_inputs).unsqueeze(0) # depth in meters (1, 300, 300)
        depth_inputs = F.interpolate(depth_inputs.unsqueeze(0), self.out_size, mode='bilinear', align_corners=True)[0]

        agent_pose = agent_pose.unsqueeze(0)
        depth_inputs = depth_inputs.unsqueeze(0)

        fx, fy, cx, cy = self.cam_params
        bs, _, imh, imw = depth_inputs.shape
        device          = depth_inputs.device

        # 2D image coordinates
        x               = rearrange(torch.arange(0, imw), 'w -> () () () w')
        y               = rearrange(torch.arange(0, imh), 'h -> () () h ()')
        x, y            = x.float().to(device), y.float().to(device)

        xx              = (x - cx) / fx
        yy              = (y - cy) / fy

        # 3D real-world coordinates (in meters)
        Z               = depth_inputs
        X               = xx * Z # (B, 1, imh, imw)
        Y               = yy * Z # (B, 1, imh, imw)

        P = torch.cat([X, Y, Z], 1) # (B, 3, imh, imw)

        P = rearrange(P, 'b p h w -> b p (h w)') # (B, 3, h*w) # matrix mult time

        # Sometimes, agent rotation is not a multiple of the interval (59 instead of 60 sometimes)
        # round them to the nearest multiple of 30
        # correct for cameraHorizon
        Rx = self.rot_x[(-agent_pose[:, 4]//self.rot_size_x).long()] # (B, 3, 3)
        Ry = self.rot_y[(agent_pose[:, 3]//self.rot_size_y).long()] # (B, 3, 3)
        R = torch.bmm(Ry, Rx) # (B, 3, 3)

        P0 = agent_pose[:, 0:3] # (B, 3)
        P0[:, 1] = -P0[:, 1] # negative y
        P0 = P0.unsqueeze(-1) # (B, 3, 1)

        R = R.to(depth_inputs.device)
        P = torch.bmm(R, P) + P0 # (B, 3, 3) * (B, 3, h*w) + (B, 3, 1) --> (B, 3, h*w)
        P = rearrange(P, 'b p (h w) -> b p h w', h=imh, w=imw)

        return 


# code below modified from: https://github.com/allenai/ai2thor/issues/124#issuecomment-473017391
class ThorPositionTo2DFrameTranslator(object):
    def __init__(self, frame_shape, cam_position, orth_size):
        self.frame_shape = frame_shape
        self.lower_left = np.array((cam_position[0], cam_position[2])) - orth_size
        self.span = 2 * orth_size

    def __call__(self, position):
        if len(position) == 3:
            x, _, z = position
        else:
            x, z = position

        camera_position = (np.array((x, z)) - self.lower_left) / self.span
        return np.array(
            (
                round(self.frame_shape[0] * (1.0 - camera_position[1])),
                round(self.frame_shape[1] * camera_position[0]),
            ),
            dtype=int,
        )

def get_points_in_fov(
    position, rotation, pos_translator, scale=1.0, opacity=0.7
):
    p0 = np.array((position[0], position[2]))
    p1 = copy.copy(p0)
    p2 = copy.copy(p0)

    theta = -2 * math.pi * (rotation / 360.0)
    rotation_mat = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    offset1 = scale * np.array([-1, 1]) * math.sqrt(2) / 2
    offset2 = scale * np.array([1, 1]) * math.sqrt(2) / 2

    p1 += np.matmul(rotation_mat, offset1)
    p2 += np.matmul(rotation_mat, offset2)

    points = [tuple(reversed(pos_translator(p))) for p in [p0, p1, p2]]

    p1, p2 = points[1], points[2]
    p = (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))

    return points + [p] # (origin, triangle pt1, triangle pt2, triangle midpoint)


def process_topdown(cam_params, pose):
    cam_position, cam_orth_size = cam_params['cam_position'], cam_params['cam_orth_size']
    cam_position = (cam_position["x"], cam_position["y"], cam_position["z"])
    pos_translator = ThorPositionTo2DFrameTranslator((300, 300), cam_position, cam_orth_size)
    points = get_points_in_fov(pose[0:3], pose[3], pos_translator)
    return points



