# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
import neural_renderer as nr

def index_selection_nd(x, I, dim):
    target_shape = [*x.shape]
    del target_shape[dim]
    target_shape[dim:dim] = [*I.shape]
    return x.index_select(dim, I.view(-1)).reshape(target_shape)

class RenderLayer(nn.Module):
    def __init__(self):
        super(RenderLayer, self).__init__()

    def forward(self, mesh, cam_param, img_affine_trans_mat, mesh):
        batch_size = mesh.shape[0]
        face_num = len(mesh['vi'])
        focal, princpt, campos, camrot = cam_param['focal'], cam_param['princpt'], cam_param['campos'], cam_param['camrot']

        # project mesh world -> camera space
        mesh = mesh - campos.view(-1,1,3)
        mesh = torch.cat([torch.mm(camrot[i],mesh[i].permute(1,0)).permute(1,0)[None,:,:] for i in range(batch_size)],0)
        mesh_3d_x = mesh[:,:,0]
        mesh_3d_y = mesh[:,:,1]
        mesh_z = mesh[:,:,2]
        mesh_z = mesh_z + (mesh_z==0).type('torch.cuda.FloatTensor')*1e-4
        
        # project camera -> image space
        mesh_2d_x = (mesh_3d_x / mesh_z * focal[:,0].view(-1,1) + princpt[:,0].view(-1,1))[:,:,None]
        mesh_2d_y = (mesh_3d_y / mesh_z * focal[:,1].view(-1,1) + princpt[:,1].view(-1,1))[:,:,None]
        mesh_2d = torch.cat([mesh_2d_x, mesh_2d_y, torch.ones_like(mesh_2d_x)],2)
        
        # apply affine transform (crop and resize)
        mesh_2d = torch.bmm(img_affine_trans_mat, mesh_2d.permute(0,2,1)).permute(0,2,1)
        
        ##################################
        # neural renderer (for depth map rendering)
        mesh_2d_norm = torch.cat([mesh_2d[:,:,0:1]/cfg.rendered_img_shape[1]*2-1,\
                (cfg.rendered_img_shape[0] - 1 - mesh_2d[:,:,1:2])/cfg.rendered_img_shape[0]*2-1, \
                mesh_z[:,:,None]],2)

        # mesh_2d_v0, v1, v2: batch_size x face_num x 3. coordinates of vertices for each face
        mesh_2d_v0 = torch.cat([index_selection_nd(mesh_2d_norm[i], mesh['vi'][:,0], 0)[None, ...] for i in range(batch_size)], dim=0)
        mesh_2d_v1 = torch.cat([index_selection_nd(mesh_2d_norm[i], mesh['vi'][:,1], 0)[None, ...] for i in range(batch_size)], dim=0)
        mesh_2d_v2 = torch.cat([index_selection_nd(mesh_2d_norm[i], mesh['vi'][:,2], 0)[None, ...] for i in range(batch_size)], dim=0)
        face_vertices = torch.cat([mesh_2d_v0[:,:,None,:], mesh_2d_v1[:,:,None,:], mesh_2d_v2[:,:,None,:]], 2)

        rendered_depthmap = nr.rasterize_depth(face_vertices, cfg.rendered_img_shape[0], False, near=cfg.depth_min, far=cfg.depth_max)[:,None,:,:]
        rendered_depthmap[rendered_depthmap == cfg.depth_max] = 0

        return rendered_depthmap

