# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.module import BackboneNet, PoseNet, SkeletonRefineNet, SkinRefineNet
#from nets.DiffableRenderer.DiffableRenderer import RenderLayer
from nets.loss import DepthmapLoss, JointLoss, PenetLoss, LaplacianLoss
from utils.transforms import euler2mat, forward_kinematics, rigid_transform_3D
from config import cfg
import math

class Model(nn.Module):
    def __init__(self, backbone_net, pose_net, skeleton_refine_net, skin_refine_net, mesh, root_joint_idx, align_joint_idx, non_rigid_joint_idx):
        super(Model, self).__init__()

        # template mesh things
        self.mesh = mesh
        self.skeleton = self.mesh.skeleton

        # keypoint things
        self.joint_num = len(self.skeleton)
        self.root_joint_idx = root_joint_idx
        self.align_joint_idx = align_joint_idx
        self.non_rigid_joint_idx = non_rigid_joint_idx

        # identity-dependent things
        self.register_buffer('id_code', torch.randn(cfg.id_code_dim))

        # modules
        self.backbone_net = backbone_net
        self.pose_net = pose_net
        self.skeleton_refine_net = skeleton_refine_net
        self.skin_refine_net = skin_refine_net
        #self.renderer = RenderLayer()
          
        # loss functions
        self.depthmap_loss = DepthmapLoss()
        self.joint_loss = JointLoss()
        self.penet_loss = PenetLoss(self.skeleton, self.mesh.segmentation, self.root_joint_idx, self.non_rigid_joint_idx)
        self.lap_loss = LaplacianLoss(self.mesh.v, self.mesh.vi)

    def get_mesh_data(self, batch_size):
        mesh  ={}
        mesh['v'] = torch.from_numpy(self.mesh.v).cuda().float()[None,:,:].repeat(batch_size,1,1) # xyz coordinates of vertex (v_xyz x 3)
        mesh['vi'] = torch.from_numpy(self.mesh.vi).cuda().long() # vertex (xyz) indices of each mesh triangle (F x 3) 
        mesh['vt'] = torch.from_numpy(self.mesh.vt).cuda().float() # uv coordinates of vertex (v_uv x 2)
        mesh['vti'] = torch.from_numpy(self.mesh.vti).cuda().long() # texture vertex (uv) indices of each mesh triangle (F x 3)
        mesh['local_pose'] = torch.from_numpy(self.mesh.local_pose).cuda().float()
        mesh['global_pose'] = torch.from_numpy(self.mesh.global_pose).cuda().float()
        mesh['global_pose_inv'] = torch.from_numpy(self.mesh.global_pose_inv).cuda().float()
        mesh['skinning_weight'] = torch.from_numpy(self.mesh.skinning_weight).cuda().float()[None,:,:].repeat(batch_size,1,1)
        mesh['segmentation'] = torch.from_numpy(self.mesh.segmentation).cuda().long()
        return mesh
    
    def forward(self, inputs, targets, meta_info, mode):
        input_img = inputs['img']
        batch_size = input_img.shape[0]
        mesh = self.get_mesh_data(batch_size)
        align_joint_idx = torch.Tensor(self.align_joint_idx).long()

        # extract image feature
        img_feat = self.backbone_net(input_img)
     
        # estimate local euler angle change for each joint
        joint_euler = self.pose_net(img_feat)
        joint_rot_mat = euler2mat(joint_euler, to_4x4=True)
        
        # estimate skeleton corrective
        skeleton_corrective = self.skeleton_refine_net(self.id_code)
        mesh['local_pose_refined'] = mesh['local_pose'].clone()
        mesh['local_pose_refined'][:,:3,3] += skeleton_corrective
        mesh['global_pose_refined'] = [None for _ in range(self.joint_num)]
        mesh['global_pose_refined'][self.root_joint_idx] = mesh['global_pose'][self.root_joint_idx].clone()
        forward_kinematics(self.skeleton, self.root_joint_idx, mesh['local_pose_refined'], mesh['global_pose_refined'])
        mesh['global_pose_refined'] = torch.stack(mesh['global_pose_refined'])

        # rigid transform for root joint
        global_pose = [[None for _ in range(self.joint_num)] for _ in range(batch_size)]
        if mode == 'train':
            for i in range(batch_size):
                cur_sample = mesh['global_pose_refined'][align_joint_idx,:3,3].view(len(align_joint_idx),3)
                gt_sample = targets['joint']['world_coord'][i][align_joint_idx].view(len(align_joint_idx),3)
                R,t = rigid_transform_3D(cur_sample, gt_sample)
                mat = torch.cat((torch.cat((R,t),1), torch.cuda.FloatTensor([[0,0,0,1]])))
                global_pose[i][self.root_joint_idx] = torch.mm(mat, mesh['global_pose_refined'][self.root_joint_idx])
        elif mode == 'test':
            for i in range(batch_size):
                global_pose[i][self.root_joint_idx] = torch.eye(4).float().cuda() # use identity matrix in testing stage


        # forward kinematics
        joint_out = []; joint_trans_mat = [];
        for i in range(batch_size):
            forward_kinematics(self.skeleton, self.root_joint_idx, torch.bmm(mesh['local_pose_refined'], joint_rot_mat[i]), global_pose[i])
            joint_out.append(torch.cat([global_pose[i][j][None,:3,3] for j in range(self.joint_num)],0))
            joint_trans_mat.append(torch.cat([torch.mm(global_pose[i][j], mesh['global_pose_inv'][j,:,:])[None,:,:] for j in range(self.joint_num)]))
        joint_out = torch.cat(joint_out).view(batch_size,self.joint_num,3)
        joint_trans_mat = torch.cat(joint_trans_mat).view(batch_size,self.joint_num,4,4).permute(1,0,2,3)
        
        # estimate corrective vector
        pose_corrective, id_corrective = self.skin_refine_net(joint_euler.detach(), self.id_code[None,:].repeat(batch_size,1))
        mesh_refined_xyz = mesh['v'] + pose_corrective + id_corrective

        # LBS
        mesh_refined_xyz1 = torch.cat([mesh_refined_xyz, torch.ones_like(mesh_refined_xyz[:,:,:1])],2)
        mesh_out_refined = sum([mesh['skinning_weight'][:,:,j,None]*torch.bmm(joint_trans_mat[j],mesh_refined_xyz1.permute(0,2,1)).permute(0,2,1)[:,:,:3] for j in range(self.joint_num)])

        # loss functions in training stage
        if mode == 'train':
            # render depthmap
            cam_param, affine_trans = meta_info['cam_param'], meta_info['affine_trans']
            depthmap_out_refined = [] 
            for cid in range(len(cam_param)):
                rendered_depthmap_refined = self.renderer(mesh_out_refined, cam_param[cid], affine_trans[cid], mesh)
                depthmap_out_refined.append(rendered_depthmap_refined)
            
            # zero pose template mesh with correctives (for penet loss and test output) 
            joint_trans_mat = torch.bmm(mesh['global_pose_refined'], mesh['global_pose_inv'])[:,None,:,:].repeat(1,batch_size,1,1)
            mesh_refined_v = mesh['v'] + pose_corrective + id_corrective
            mesh_refined_v = torch.cat([mesh_refined_v, torch.ones_like(mesh_refined_v[:,:,:1])],2)
            mesh_refined_v = sum([mesh['skinning_weight'][:,:,j,None]*torch.bmm(joint_trans_mat[j],mesh_refined_v.permute(0,2,1)).permute(0,2,1)[:,:,:3] for j in range(self.joint_num)])

            loss = {}
            loss['joint'] = self.joint_loss(joint_out, targets['joint']['world_coord'], targets['joint']['valid'])
            loss['depthmap'] = self.depthmap_loss(depthmap_out_refined, targets['depthmap'])
            loss['penet'] = self.penet_loss(mesh['global_pose_refined'].detach(), mesh_refined_v.detach(), joint_out, mesh_out_refined)
            loss['lap'] = self.lap_loss(mesh_out_refined) * cfg.loss_lap_weight
            return loss
        
        # output in testing stage
        elif mode == 'test':
            out = {}
            out['joint_out'] = joint_out
            out['mesh_out_refined'] = mesh_out_refined
            return out

    def decode(self, joint_euler):
        batch_size = joint_euler.shape[0]
        mesh = self.get_mesh_data(batch_size)
        joint_rot_mat = euler2mat(joint_euler, to_4x4=True)

        # estimate skeleton corrective
        skeleton_corrective = self.skeleton_refine_net(self.id_code)
        mesh['local_pose_refined'] = mesh['local_pose'].clone()
        mesh['local_pose_refined'][:,:3,3] += skeleton_corrective
        mesh['global_pose_refined'] = [None for _ in range(self.joint_num)]
        mesh['global_pose_refined'][self.root_joint_idx] = mesh['global_pose'][self.root_joint_idx].clone()
        forward_kinematics(self.skeleton, self.root_joint_idx, mesh['local_pose_refined'], mesh['global_pose_refined'])
        mesh['global_pose_refined'] = torch.stack(mesh['global_pose_refined'])

        # rigid transform for root joint
        global_pose = [[None for _ in range(self.joint_num)] for _ in range(batch_size)]
        for i in range(batch_size):
            global_pose[i][self.root_joint_idx] = torch.eye(4).float().cuda() # use identity matrix in testing stage

        # forward kinematics
        joint_out = []; joint_trans_mat = [];
        for i in range(batch_size):
            forward_kinematics(self.skeleton, self.root_joint_idx, torch.bmm(mesh['local_pose_refined'], joint_rot_mat[i]), global_pose[i])
            joint_out.append(torch.cat([global_pose[i][j][None,:3,3] for j in range(self.joint_num)],0))
            joint_trans_mat.append(torch.cat([torch.mm(global_pose[i][j], mesh['global_pose_inv'][j,:,:])[None,:,:] for j in range(self.joint_num)]))
        joint_out = torch.cat(joint_out).view(batch_size,self.joint_num,3)
        joint_trans_mat = torch.cat(joint_trans_mat).view(batch_size,self.joint_num,4,4).permute(1,0,2,3)
        
        # estimate corrective vector
        pose_corrective, id_corrective = self.skin_refine_net(joint_euler.detach(), self.id_code[None,:].repeat(batch_size,1))
        mesh_refined_xyz = mesh['v'] + pose_corrective + id_corrective

        # LBS
        mesh_refined_xyz1 = torch.cat([mesh_refined_xyz, torch.ones_like(mesh_refined_xyz[:,:,:1])],2)
        mesh_out_refined = sum([mesh['skinning_weight'][:,:,j,None]*torch.bmm(joint_trans_mat[j],mesh_refined_xyz1.permute(0,2,1)).permute(0,2,1)[:,:,:3] for j in range(self.joint_num)])

        out = {}
        out['joint_out'] = joint_out
        out['mesh_out_refined'] = mesh_out_refined
        return out

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(mode, mesh, root_joint_idx, align_joint_idx, non_rigid_joint_idx):
    backbone_net = BackboneNet()
    pose_net = PoseNet(mesh.skeleton)
    skeleton_refine_net = SkeletonRefineNet(mesh.skeleton)
    skin_refine_net = SkinRefineNet(mesh.skeleton, len(mesh.v))

    if mode == 'train':
        backbone_net.init_weights()
        pose_net.apply(init_weights)
        skeleton_refine_net.apply(init_weights)
        skin_refine_net.apply(init_weights)

    model = Model(backbone_net, pose_net, skeleton_refine_net, skin_refine_net, mesh, root_joint_idx, align_joint_idx, non_rigid_joint_idx)
    return model

