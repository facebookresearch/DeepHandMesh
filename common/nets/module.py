# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from torch.nn import functional as F
from config import cfg
from nets.layer import make_linear_layers, make_conv_layers, make_deconv_layers
from nets.resnet import ResNetBackbone
import math

class BackboneNet(nn.Module):
    def __init__(self):
        super(BackboneNet, self).__init__()
        self.resnet = ResNetBackbone(cfg.resnet_type)
        self.fc = make_linear_layers([2048,cfg.backbone_img_feat_dim])
   
    def init_weights(self):
        self.resnet.init_weights()

    def forward(self, img):
        img_feat = self.resnet(img)
        img_feat = F.avg_pool2d(img_feat,(img_feat.shape[2],img_feat.shape[3])).view(-1,2048)
        img_feat = self.fc(img_feat)
        return img_feat

class PoseNet(nn.Module):
    def __init__(self, skeleton):
        super(PoseNet, self).__init__()
        self.joint_num = len(skeleton)
        self.register_buffer('DoF', torch.cat([torch.from_numpy(skeleton[i]['DoF'])[None,:] for i in range(self.joint_num)]).cuda().float())
        self.dof_num = int(torch.sum(self.DoF))
        self.fc = make_linear_layers([cfg.backbone_img_feat_dim,512,self.dof_num], relu_final=False)

    def output_to_euler_angle(self, x):
        batch_size = x.shape[0]
        idx = torch.nonzero(self.DoF) # idx[:,0]: joint_idx, idx[:,1]: 0: x, 1: y, 2: z
        
        # normalize x to [-1,1]
        x = torch.tanh(x) * math.pi
        
        # plug in estimated euler angle. for DoF==0, angles are to zero
        euler_angle = torch.zeros((batch_size, self.joint_num, 3)).cuda().float()
        euler_angle[:,idx[:,0],idx[:,1]] = x

        return euler_angle

    def forward(self, img_feat_all_view):
        output = self.fc(img_feat_all_view)
        angle = self.output_to_euler_angle(output)
        return angle

class SkeletonRefineNet(nn.Module):
    def __init__(self, skeleton):
        super(SkeletonRefineNet, self).__init__()
        self.joint_num = len(skeleton)
        self.fc_skeleton = make_linear_layers([cfg.id_code_dim,64,self.joint_num*3], relu_final=False)
    
    def forward(self, id_code):
        skeleton_corrective = self.fc_skeleton(id_code).view(self.joint_num,3)
        return skeleton_corrective

class SkinRefineNet(nn.Module):
    def __init__(self, skeleton, vertex_num):
        super(SkinRefineNet, self).__init__()
        self.joint_num = len(skeleton)
        self.vertex_num = vertex_num
        self.fc_pose = make_linear_layers([self.joint_num*3,256,vertex_num*3], relu_final=False)
        self.fc_id = make_linear_layers([cfg.id_code_dim,256,vertex_num*3], relu_final=False)

    def forward(self, joint_euler, id_code):
        joint_euler = joint_euler.view(-1,self.joint_num*3)
        pose_corrective = self.fc_pose(joint_euler).view(-1,self.vertex_num,3)
        id_corrective = self.fc_id(id_code).view(-1,self.vertex_num,3)
        return pose_corrective, id_corrective


