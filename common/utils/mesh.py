# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from config import cfg
import os
import os.path as osp
import math
from utils.preprocessing import load_img
import cv2

class Mesh(object):
    # A simple class for creating and manipulating trimesh objects
    def __init__(self, obj_filename):

        self.v = np.zeros((0,3), dtype=np.float32) # (x,y,z) coordinates of mesh
        self.vi = np.zeros((0,3), dtype=np.int32) # vertex indices (v) of each mesh triangle 
        self.vt = np.zeros((0,2), dtype=np.float32) # (u,v) coordinates of texture in UV space
        self.vti = np.zeros((0,3), dtype=np.int32) # vertex indices (vt) of each mesh triangle
        self.load_obj(obj_filename)
        
        self.skeleton= None # joint information 
        self.global_pose = None # 6D global pose
        self.skinning_weight = None # skinning weight of template mesh
        self.segmentation = None # segmentation obtained from skinnig weight
        self.local_pose = None # local pose
        self.global_pose_inv = None # inverse of global pose
        
    def load_obj(self, file_name):
        obj_file = open(file_name)
        for line in obj_file:
            words = line.split(' ')
            if words[0] == 'v':
                x,y,z = float(words[1]) * 10, float(words[2]) * 10, float(words[3]) * 10 # change cm to mm
                self.v = np.concatenate((self.v, np.array([x,y,z]).reshape(1,3)), axis=0)
            elif words[0] == 'vt':
                u,v = float(words[1]), float(words[2])
                self.vt = np.concatenate((self.vt, np.array([u,v]).reshape(1,2)), axis=0)
            elif words[0] == 'f':
                vi_1, vti_1 = words[1].split('/')[:2]
                vi_2, vti_2 = words[2].split('/')[:2]
                vi_3, vti_3 = words[3].split('/')[:2]
 
                # change 1-based index to 0-based index
                vi_1, vi_2, vi_3 = int(vi_1)-1, int(vi_2)-1, int(vi_3)-1 
                vti_1, vti_2, vti_3 = int(vti_1)-1, int(vti_2)-1, int(vti_3)-1

                self.vi = np.concatenate((self.vi, np.array([vi_1,vi_2,vi_3]).reshape(1,3)), axis=0)
                self.vti = np.concatenate((self.vti, np.array([vti_1,vti_2,vti_3]).reshape(1,3)), axis=0)
            else:
                pass
    
    def save_obj(self, v, color=None, file_name='output.obj'):
        obj_file = open(file_name, 'w')
        for i in range(len(v)):
            if color is None:
                obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
            else:
                obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + ' ' + str(color[i][0]) + ' ' + str(color[i][1]) + ' ' + str(color[i][2]) + '\n')
        for i in range(len(self.vt)):
            obj_file.write('vt ' + str(self.vt[i][0]) + ' ' + str(self.vt[i][1]) + '\n')
        for i in range(len(self.vi)):
            obj_file.write('f ' + str(self.vi[i][0]+1) + '/' + str(self.vti[i][0]+1) + ' ' + str(self.vi[i][1]+1) + '/' + str(self.vti[i][1]+1) + ' ' + str(self.vi[i][2]+1) + '/' + str(self.vti[i][2]+1) + '\n')
        obj_file.close()
    
    def load_skeleton(self, path, joint_num):

        # load joint info (name, parent_id)
        skeleton = [{} for _ in range(joint_num)]
        with open(path) as fp:
            for line in fp:
                if line[0] == '#': continue
                splitted = line.split(' ')
                joint_name, joint_id, joint_parent_id = splitted
                joint_id, joint_parent_id = int(joint_id), int(joint_parent_id)
                skeleton[joint_id]['name'] = joint_name
                skeleton[joint_id]['parent_id'] = joint_parent_id
                 
                if joint_name.endswith('null'):
                    skeleton[joint_id]['DoF'] = np.array([0,0,0],dtype=np.float32)
                elif joint_name.endswith('3'):
                    skeleton[joint_id]['DoF'] = np.array([0,0,1],dtype=np.float32)
                elif joint_name.endswith('2'):
                    skeleton[joint_id]['DoF'] = np.array([0,0,1],dtype=np.float32)
                elif joint_name.endswith('1'):
                    skeleton[joint_id]['DoF'] = np.array([1,1,1],dtype=np.float32)
                elif joint_name.endswith('thumb0'):
                    skeleton[joint_id]['DoF'] = np.array([1,1,1],dtype=np.float32)
                else:
                    skeleton[joint_id]['DoF'] = np.array([0,0,0],dtype=np.float32)

        # save child_id
        for i in range(len(skeleton)):
            joint_child_id = []
            for j in range(len(skeleton)):
                if skeleton[j]['parent_id'] == i:
                    joint_child_id.append(j)
            skeleton[i]['child_id'] = joint_child_id
        
        self.skeleton = skeleton

    def load_skinning_weight(self, skinning_weight_path):
        # load skinning_weight of mesh
        self.skinning_weight = np.zeros((len(self.v),len(self.skeleton)), dtype=np.float32)
        self.segmentation = np.zeros((len(self.v)), dtype=np.float32)
        with open(skinning_weight_path) as fp:
            for line in fp:
                if line[0] == '#': continue
                splitted = line.split(' ')[:-1]
                vertex_idx = int(splitted[0])
                for i in range(0,len(splitted)-1,2):
                    joint_name = splitted[1+i]
                    joint_idx = [idx for idx,_ in enumerate(self.skeleton) if _['name'] == joint_name][0]
                    weight = float(splitted[1+(i+1)])
                    self.skinning_weight[vertex_idx][joint_idx] = weight

                self.segmentation[vertex_idx] = np.argmax(self.skinning_weight[vertex_idx])

    def load_global_pose(self, global_pose_path):
        # load global pose of mesh
        self.global_pose = np.zeros((len(self.skeleton),4,4), dtype=np.float32)
        with open(global_pose_path) as fp:
            for line in fp:
                if line[0] == '#': continue
                splitted = line.split(' ')
                joint_name = splitted[0]
                joint_idx = [i for i,_ in enumerate(self.skeleton) if _['name'] == joint_name][0]
                self.global_pose[joint_idx] = np.array([float(x) for x in splitted[1:-1]]).reshape(4,4)

    def load_local_pose(self, local_pose_path):
        # load local pose of mesh
        self.local_pose = np.zeros((len(self.skeleton),4,4), dtype=np.float32)
        with open(local_pose_path) as fp:
            for line in fp:
                if line[0] == '#': continue
                splitted = line.split(' ')
                joint_name = splitted[0]
                joint_idx = [i for i,_ in enumerate(self.skeleton) if _['name'] == joint_name][0]
                self.local_pose[joint_idx] = np.array([float(x) for x in splitted[1:-1]]).reshape(4,4)
    
    def load_global_pose_inv(self, global_pose_inv_path):
        # load global inv pose of mesh
        self.global_pose_inv = np.zeros((len(self.skeleton),4,4), dtype=np.float32)
        with open(global_pose_inv_path) as fp:
            for line in fp:
                if line[0] == '#': continue
                splitted = line.split(' ')
                joint_name = splitted[0]
                joint_idx = [i for i,_ in enumerate(self.skeleton) if _['name'] == joint_name][0]
                self.global_pose_inv[joint_idx] = np.array([float(x) for x in splitted[1:-1]]).reshape(4,4)
    

