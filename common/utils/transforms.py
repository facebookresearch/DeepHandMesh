# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
from config import cfg

def cam2pixel(cam_coord, f, c):
    x = cam_coord[..., 0] / (cam_coord[..., 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[..., 1] / (cam_coord[..., 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[..., 2]
    return x,y,z

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[..., 0] - c[0]) / f[0] * pixel_coord[..., 2]
    y = (pixel_coord[..., 1] - c[1]) / f[1] * pixel_coord[..., 2]
    z = pixel_coord[..., 2]
    return x,y,z

def world2cam(world_coord, R, T):
    cam_coord = np.dot(R, world_coord - T)
    return cam_coord

def forward_kinematics(skeleton, cur_joint_idx, local_pose, global_pose):
        
    child_id = skeleton[cur_joint_idx]['child_id']
    if len(child_id) == 0:
        return
    
    for joint_id in child_id:
        global_pose[joint_id] = torch.mm(global_pose[cur_joint_idx], local_pose[joint_id])
        forward_kinematics(skeleton, joint_id, local_pose, global_pose)

def rigid_transform_3D(A, B):
    centroid_A = torch.mean(A,0)
    centroid_B = torch.mean(B,0)
    H = torch.mm((A - centroid_A).permute(1,0), B - centroid_B)

    U, s, V = torch.svd(H)

    R = torch.mm(V, U.permute(1,0))
    if torch.det(R) < 0:
        V = torch.stack((V[:,0],V[:,1],-V[:,2]),1)
        #V[:,2] = -V[:,2]
        R = torch.mm(V, U.permute(1,0))
    t = -torch.mm(R, centroid_A[:,None]) + centroid_B[:,None]
    t = t.view(3,1)
    return R, t

def euler2mat(theta, to_4x4=False):
     
    assert theta.shape[-1] == 3

    original_shape = list(theta.shape)
    original_shape.append(3)

    theta = theta.view(-1, 3)
    theta_x = theta[:,0:1]
    theta_y = theta[:,1:2]
    theta_z = theta[:,2:3]
    
    R_x = torch.cat([\
            torch.stack([torch.ones_like(theta_x), torch.zeros_like(theta_x), torch.zeros_like(theta_x)],2),\
            torch.stack([torch.zeros_like(theta_x), torch.cos(theta_x), -torch.sin(theta_x)],2),\
            torch.stack([torch.zeros_like(theta_x), torch.sin(theta_x), torch.cos(theta_x)],2)\
            ],1)

    R_y = torch.cat([\
            torch.stack([torch.cos(theta_y), torch.zeros_like(theta_y), torch.sin(theta_y)],2),\
            torch.stack([torch.zeros_like(theta_y), torch.ones_like(theta_y), torch.zeros_like(theta_y)],2),\
            torch.stack([-torch.sin(theta_y), torch.zeros_like(theta_y), torch.cos(theta_y)],2),\
            ],1)
    
    R_z = torch.cat([\
            torch.stack([torch.cos(theta_z), -torch.sin(theta_z), torch.zeros_like(theta_z)],2),\
            torch.stack([torch.sin(theta_z), torch.cos(theta_z), torch.zeros_like(theta_z)],2),\
            torch.stack([torch.zeros_like(theta_z), torch.zeros_like(theta_z), torch.ones_like(theta_z)],2),\
            ],1)
                     
    R = torch.bmm(R_z, torch.bmm( R_y, R_x ))

    if to_4x4:
        batch_size = R.shape[0]
        R = torch.cat([R,torch.zeros((batch_size,3,1)).cuda().float()],2)
        R = torch.cat([R,torch.cuda.FloatTensor([0,0,0,1])[None,None,:].repeat(batch_size,1,1)],1) # 0001
        original_shape[-2] = 4; original_shape[-1] = 4

    R = R.view(original_shape)
    return R

def rgb2gray(rgb):
    r,g,b = rgb[0], rgb[1], rgb[2]
    gray = 0.2989*r + 0.5870*g + 0.1140*b
    return gray
