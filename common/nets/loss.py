# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from config import cfg
import math

class DepthmapLoss(nn.Module):
    def __init__(self):
        super(DepthmapLoss, self).__init__()
    
    def smooth_l1_loss(self, out, gt):
        return F.smooth_l1_loss(out, gt, reduction='none')

    def forward(self, depthmap_out, depthmap_gt):
        if isinstance(depthmap_out,list) and isinstance(depthmap_gt,list):
            mask = [((depthmap_out[i] != 0) * (depthmap_gt[i] != 0)).float() for i in range(len(depthmap_out))]
            loss = [self.smooth_l1_loss(depthmap_out[i], depthmap_gt[i]) * mask[i] for i in range(len(depthmap_out))]
            loss = sum(loss)/len(loss)
        elif isinstance(depthmap_out,torch.Tensor) and isinstance(depthmap_gt,torch.Tensor):
            mask = ((depthmap_out != 0) * (depthmap_gt != 0)).float()
            loss = self.smooth_l1_loss(depthmap_out, depthmap_gt) * mask
        else:
            assert 0

        return loss

class JointLoss(nn.Module):
    def __init__(self):
        super(JointLoss, self).__init__()

    def l1_loss(self, out, gt):
        return torch.abs(out - gt)

    def forward(self, joint_out, joint_gt, joint_valid):
        loss = self.l1_loss(joint_out, joint_gt) * joint_valid
        return loss

class PenetLoss(nn.Module):
    def __init__(self, skeleton, segmentation, root_joint_idx, non_rigid_joint_idx):
        super(PenetLoss, self).__init__()
        self.skeleton = skeleton
        self.joint_num = len(skeleton)
        self.root_joint_idx = root_joint_idx
        self.non_rigid_joint_idx = non_rigid_joint_idx
        self.register_buffer('segmentation', torch.from_numpy(segmentation).cuda().float())

    def traverse_skeleton(self, cur_joint_idx, path, all_path):
        # skeleton hierarchy
        path.append(cur_joint_idx)
        if len(self.skeleton[cur_joint_idx]['child_id']) > 0:
            for child_id in self.skeleton[cur_joint_idx]['child_id']:
                path_for_each_node = path.copy()
                self.traverse_skeleton(child_id, path, all_path)
                path = path_for_each_node
        else:
            all_path.append(path)

    def make_combination(self, data):
        combination = []
        assert len(data) > 1
        for i in range(len(data)-1):
            for j in range(i+1,len(data)):
                combination.append([data[i],data[j]])
        return combination

    def make_bone_helper(self, bone, template_pose, mesh_v, start_joint_idx, end_joint_idx, step):
        bone['start_joint_idx'].append(start_joint_idx)
        bone['end_joint_idx'].append(end_joint_idx)
        bone['step'].append(step)
        bone['point_num'] += 1
        
        cur_dir = template_pose[end_joint_idx,:3,3] - template_pose[start_joint_idx,:3,3]
        cur_pos = template_pose[start_joint_idx,:3,3] + cur_dir * step
        vector = mesh_v - cur_pos[None,None,:]
        min_dist, min_dist_idx = torch.min(torch.sqrt(torch.sum(vector**2,2)),1)#[0]
        bone['min_dist'].append(min_dist)

    def make_bone(self, bone, template_pose, mesh_v, cur_joint_idx):
        # make bone by interpolating joint of mesh
        if len(self.skeleton[cur_joint_idx]['child_id']) == 0:
            self.make_bone_helper(bone, template_pose, mesh_v, cur_joint_idx, cur_joint_idx, 0)

        for child_joint_idx in self.skeleton[cur_joint_idx]['child_id']:
            for step in torch.arange(0,1,cfg.bone_step_size):
                self.make_bone_helper(bone, template_pose, mesh_v, cur_joint_idx, child_joint_idx, step)
            self.make_bone(bone, template_pose, mesh_v, child_joint_idx)

    def get_bone_data(self, template_pose, mesh_v):
        bone = {'start_joint_idx': [], 'end_joint_idx': [], 'step': [], 'min_dist': [], 'point_num': 0}
        self.make_bone(bone, template_pose, mesh_v, self.root_joint_idx)
        bone['start_joint_idx'] = torch.cuda.LongTensor(bone['start_joint_idx'])
        bone['end_joint_idx'] = torch.cuda.LongTensor(bone['end_joint_idx'])
        bone['step'] = torch.cuda.FloatTensor(bone['step'])
        bone['min_dist'] = torch.stack(bone['min_dist']) # point_num x batch_size
        return bone

    def get_bone_from_joint(self, joint, bone):
        batch_size = joint.shape[0]
        bone_num = bone['point_num']
        vec_dir = joint[:,bone['end_joint_idx'],:] - joint[:,bone['start_joint_idx'],:]
        bone_out = joint[:,bone['start_joint_idx'],:] + vec_dir * bone['step'][None,:,None].repeat(batch_size,1,3)
        bone_out = bone_out.view(batch_size,bone_num,3)
        return bone_out
    
    def forward(self, template_pose, mesh_v, joint_out, geo_out):
        batch_size = joint_out.shape[0]
    
        bone = self.get_bone_data(template_pose, mesh_v)
        bone_out_from_joint = self.get_bone_from_joint(joint_out, bone)

        skeleton_path = []
        self.traverse_skeleton(self.root_joint_idx, [], skeleton_path)
        skeleton_part = []
        for path in skeleton_path:
            for pid in range(len(path)-1):
                start_joint_idx = path[pid]; end_joint_idx = path[pid+1]
                skeleton_part.append([start_joint_idx, end_joint_idx])
        skeleton_part = self.make_combination(skeleton_part) # (combination num x 2 (part_1, part_2) x 2 (start, end joint idx))
        
        # rigid part
        loss_penetration_rigid = 0
        loss_penetration_rigid_cnt = 0
        for cid in range(len(skeleton_part)):
            # first part index
            start_joint_idx_1 = skeleton_part[cid][0][0]
            end_joint_idx_1 = skeleton_part[cid][0][1]
 
            # second part index
            start_joint_idx_2 = skeleton_part[cid][1][0]
            end_joint_idx_2 = skeleton_part[cid][1][1]
            
            # exclude adjant parts
            if start_joint_idx_1 == start_joint_idx_2 or start_joint_idx_1 == end_joint_idx_2 or end_joint_idx_1 == start_joint_idx_2:
                continue
            
            # first part distance thr (radius of sphere)
            bone_mask_1 = ((bone['start_joint_idx'] == start_joint_idx_1) * (bone['end_joint_idx'] == end_joint_idx_1)).byte()
            if torch.sum(bone_mask_1) == 0:
                continue
            bone_1 = bone_out_from_joint[:,bone_mask_1,:]
            dist_thr_1 = bone['min_dist'][bone_mask_1].permute(1,0)

            # second part distance thr (radius of sphere)
            bone_mask_2 = ((bone['start_joint_idx'] == start_joint_idx_2) * (bone['end_joint_idx'] == end_joint_idx_2)).byte()
            if torch.sum(bone_mask_2) == 0:
                continue
            bone_2 = bone_out_from_joint[:,bone_mask_2,:]
            dist_thr_2 = bone['min_dist'][bone_mask_2].permute(1,0)

            # loss calculate
            dist = torch.sqrt(torch.sum((bone_1[:,:,None,:].repeat(1,1,bone_2.shape[1],1) - bone_2[:,None,:,:].repeat(1,bone_1.shape[1],1,1))**2,3))
            dist_thr = dist_thr_1[:,:,None].repeat(1,1,dist_thr_2.shape[1]) + dist_thr_2[:,None,:].repeat(1,dist_thr_1.shape[1],1)
            loss_penetration_rigid += torch.clamp(dist_thr - dist, min=0).mean((1,2))
            loss_penetration_rigid_cnt += 1

        # non-rigid joint
        loss_penetration_non_rigid = 0
        loss_penetration_non_rigid_cnt = 0
        for nr_jid in self.non_rigid_joint_idx:
            nr_skin = geo_out[:,(self.segmentation == nr_jid).byte(),:]
            for path in skeleton_path:
                is_penetrating = torch.cuda.FloatTensor([0 for _ in range(batch_size)])

                # only consider finger tips
                for pid in range(len(path)-2,len(path)-1):
                    start_joint_idx = path[pid]; end_joint_idx = path[pid+1];

                    # exclude path from root through the thumb path
                    if 'thumb' in self.skeleton[start_joint_idx]['name'] or 'thumb' in self.skeleton[end_joint_idx]['name']:
                        continue

                    bone_mask = ((bone['start_joint_idx'] == start_joint_idx) * (bone['end_joint_idx'] == end_joint_idx)).byte()
                    bone_pid = bone_out_from_joint[:,bone_mask,:]
                    dist = torch.sqrt(torch.sum((nr_skin[:,:,None,:].repeat(1,1,bone_pid.shape[1],1) - bone_pid[:,None,:,:].repeat(1,nr_skin.shape[1],1,1))**2,3))
                    dist = torch.min(dist,1)[0] # use minimum distance from a bone to skin
                    dist_thr = bone['min_dist'][bone_mask].permute(1,0)
                    
                    loss_per_batch = []
                    for bid in range(batch_size):
                        collision_idx = torch.nonzero(dist[bid] < dist_thr[bid])
                        if len(collision_idx) > 0: # collision occur
                            is_penetrating[bid] = 1
                            bone_idx = torch.min(collision_idx) # bone start from parent to child -> just pick min idx
                            loss = torch.abs((dist[bid][bone_idx:] - dist_thr[bid][bone_idx:]).mean()).view(1)
                        elif is_penetrating[bid] == 1:
                            loss = torch.abs((dist[bid] - dist_thr[bid]).mean()).view(1)
                        else:
                            loss = torch.zeros((1)).cuda().float()
                        loss_per_batch.append(loss)
                    loss_penetration_non_rigid += torch.cat(loss_per_batch)
                    loss_penetration_non_rigid_cnt += 1

        loss_penetration_rigid = loss_penetration_rigid / loss_penetration_rigid_cnt
        loss_penetration_non_rigid = loss_penetration_non_rigid / loss_penetration_non_rigid_cnt
        loss = cfg.loss_penet_r_weight * loss_penetration_rigid + cfg.loss_penet_nr_weight * loss_penetration_non_rigid
        return loss

class LaplacianLoss(nn.Module):
    def __init__(self, vertex, faces, average=False):
        super(LaplacianLoss, self).__init__()
        self.nv = vertex.shape[0]
        self.nf = faces.shape[0]
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)
        
        laplacian[faces[:, 0], faces[:, 1]] = -1
        laplacian[faces[:, 1], faces[:, 0]] = -1
        laplacian[faces[:, 1], faces[:, 2]] = -1
        laplacian[faces[:, 2], faces[:, 1]] = -1
        laplacian[faces[:, 2], faces[:, 0]] = -1
        laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(self.nv):
            laplacian[i, :] /= (laplacian[i, i] + 1e-8)

        self.register_buffer('laplacian', torch.from_numpy(laplacian).cuda().float())

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.cat([torch.matmul(self.laplacian,x[i])[None,:,:] for i in range(batch_size)],0)
        x = x.pow(2).sum(2)
        if self.average:
            return x.sum() / batch_size
        else:
            return x
        
