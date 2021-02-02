# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import torch.utils.data
import cv2
from glob import glob
import os.path as osp
from config import cfg
from utils.preprocessing import load_img, load_krt, get_bbox, generate_patch_image, gen_trans_from_patch_cv
from utils.transforms import world2cam, cam2pixel
from utils import mesh
import pickle
from PIL import Image, ImageDraw
import random

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.root_path = '../data/data'
        self.depthmap_root_path = '../data/depthmap'
        self.hand_model_path = '../data/hand_model'

        self.sequence_names = glob(osp.join(self.root_path, '*RT*')) # hand v2. RT: right single hand, LT: left single hand, DH: double hands
        self.sequence_names = [name for name in self.sequence_names if osp.isdir(name)]
        self.sequence_names = [name for name in self.sequence_names if 'ROM' not in name] # 'ROM' exclude in training
        self.sequence_names = [x.split('/')[-1] for x in self.sequence_names]

        self.krt_path = osp.join(self.root_path, 'KRT') # camera parameters

        self.obj_file_path = osp.join(self.hand_model_path, 'hand.obj')
        self.template_skeleton_path = osp.join(self.hand_model_path, 'skeleton.txt')
        self.template_skinning_weight_path = osp.join(self.hand_model_path, 'skinning_weight.txt')
        self.template_local_pose_path = osp.join(self.hand_model_path, 'local_pose.txt')
        self.template_global_pose_path = osp.join(self.hand_model_path, 'global_pose.txt')
        self.template_global_pose_inv_path = osp.join(self.hand_model_path, 'global_pose_inv.txt')
 
        self.joint_num = 22
        self.root_joint_idx = 21
        self.align_joint_idx = [8, 12, 16, 20, 21]
        self.non_rigid_joint_idx = [3,21]

        self.original_img_shape = (512, 334) # downsized from (4096, 2668). (height, width)
        self.transform = transform
        
        # load template mesh things
        self.mesh = mesh.Mesh(self.obj_file_path) 
        self.mesh.load_skeleton(self.template_skeleton_path, self.joint_num)
        self.mesh.load_skinning_weight(self.template_skinning_weight_path) 
        self.mesh.load_local_pose(self.template_local_pose_path) 
        self.mesh.load_global_pose(self.template_global_pose_path) 
        self.mesh.load_global_pose_inv(self.template_global_pose_inv_path) 

        # camera load
        krt = load_krt(self.krt_path)
        self.all_cameras = krt.keys()
        self.selected_cameras = [x for x in self.all_cameras if x[:2] == "40"] # 40xx cameras: color cameras
 
        # compute view directions of each camera
        campos = {}
        camrot = {}
        focal = {}
        princpt = {}
        for cam in self.selected_cameras:
            campos[cam] = -np.dot(krt[cam]['extrin'][:3, :3].T, krt[cam]['extrin'][:3, 3]).astype(np.float32)
            camrot[cam] = np.array(krt[cam]['extrin'][:3, :3]).astype(np.float32)
            focal[cam] = krt[cam]['intrin'][:2, :2]
            focal[cam] = np.array([focal[cam][0][0], focal[cam][1][1]]).astype(np.float32) / 8 # downsized from 4K to 512
            princpt[cam] = np.array(krt[cam]['intrin'][:2, 2]).astype(np.float32) / 8 # downsized from 4K to 512
        self.campos = campos
        self.camrot = camrot
        self.focal = focal
        self.princpt = princpt
      
        # get info for all frames
        self.framelist = []
        for seq_name in self.sequence_names:
            frameinfo = []
            with open(osp.join(self.root_path, seq_name,'frame')) as f:
                frameinfo = f.readline().split()
            start_frame = int(frameinfo[2])
            end_frame = int(frameinfo[3])
            frame_interval = int(frameinfo[4])
            for cam in self.selected_cameras:
                for frame_idx in range(start_frame, end_frame+1, frame_interval):
                   
                    # load joint world coordinates
                    joint_path = osp.join(self.root_path, seq_name, 'keypoints', '3D', 'image' + "{:04d}".format(frame_idx) + '.pts')
                    joint_coord, joint_valid = self.load_joint_coord(joint_path, 'right', self.mesh.skeleton)
       
                    joint = {'world_coord': joint_coord, 'valid': joint_valid}
                    frame = {'seq_name': seq_name, 'cam': cam, 'frame_idx': frame_idx, 'joint': joint}
                    self.framelist.append(frame)


    def __len__(self):
        return len(self.framelist)
    
    def __getitem__(self, idx):
        frame = self.framelist[idx]
        seq_name, cam, frame_idx, joint = frame['seq_name'], frame['cam'], frame['frame_idx'], frame['joint']
        joint_coord, joint_valid = joint['world_coord'], joint['valid']
       
        # input data
        # bbox calculate
        bbox = get_bbox(joint_coord, joint_valid, self.camrot[cam], self.campos[cam], self.focal[cam], self.princpt[cam])
        xmin, ymin, xmax, ymax = bbox
        xmin = max(xmin,0); ymin = max(ymin,0); xmax = min(xmax, self.original_img_shape[1]-1); ymax = min(ymax, self.original_img_shape[0]-1);
        bbox = np.array([xmin, ymin, xmax, ymax])
        
        # image read
        img_path = osp.join(self.root_path, seq_name, 'images', 'cam' + cam, 'image' + "{:04d}".format(frame_idx) + '.png')
        img = load_img(img_path)
        xmin, ymin, xmax, ymax = bbox
        xmin, xmax = np.array([xmin, xmax])/self.original_img_shape[1]*img.shape[1]; ymin, ymax = np.array([ymin, ymax])/self.original_img_shape[0]*img.shape[0]
        bbox_img = np.array([xmin, ymin, xmax-xmin+1, ymax-ymin+1])
        img = generate_patch_image(img, bbox_img, False, 1.0, 0.0, cfg.input_img_shape)
        input_img = self.transform(img)/255.


        target_depthmaps = []; cam_params = []; affine_transes = [];
        for cam in random.sample(self.selected_cameras, cfg.render_view_num):
            # bbox calculate
            bbox = get_bbox(joint_coord, joint_valid, self.camrot[cam], self.campos[cam], self.focal[cam], self.princpt[cam])
            xmin, ymin, xmax, ymax = bbox
            xmin = max(xmin,0); ymin = max(ymin,0); xmax = min(xmax, self.original_img_shape[1]-1); ymax = min(ymax, self.original_img_shape[0]-1);
            bbox = np.array([xmin, ymin, xmax, ymax])

            # depthmap read
            depthmap_path = osp.join(self.depthmap_root_path, "{:06d}".format(frame_idx), 'depthmap' + cam + '.pkl')
            with open(depthmap_path,'rb') as f:
                depthmap = pickle.load(f).astype(np.float32)
            xmin, ymin, xmax, ymax = bbox
            xmin, xmax = np.array([xmin, xmax])/self.original_img_shape[1]*depthmap.shape[1]; ymin, ymax = np.array([ymin, ymax])/self.original_img_shape[0]*depthmap.shape[0]
            bbox_depthmap = np.array([xmin, ymin, xmax-xmin+1, ymax-ymin+1])
            depthmap = generate_patch_image(depthmap[:,:,None], bbox_depthmap, False, 1.0, 0.0, cfg.rendered_img_shape)
            target_depthmaps.append(self.transform(depthmap))

            xmin, ymin, xmax, ymax = bbox
            affine_transes.append(gen_trans_from_patch_cv((xmin+xmax+1)/2., (ymin+ymax+1)/2., xmax-xmin+1, ymax-ymin+1, cfg.rendered_img_shape[1], cfg.rendered_img_shape[0], 1.0, 0.0).astype(np.float32))
            cam_params.append({'camrot': self.camrot[cam], 'campos': self.campos[cam], 'focal': self.focal[cam], 'princpt': self.princpt[cam]})
        
        inputs = {'img': input_img}
        targets = {'depthmap': target_depthmaps, 'joint': joint}
        meta_info = {'cam_param': cam_params, 'affine_trans': affine_transes}
      
        return inputs, targets, meta_info

    def load_joint_coord(self, joint_path, hand_type, skeleton):

        # create link between (joint_index in file, joint_name)
        # all the codes use joint index and name of 'skeleton'
        db_joint_name = ['b_r_thumb_null', 'b_r_thumb3', 'b_r_thumb2', 'b_r_thumb1', 'b_r_index_null', 'b_r_index3', 'b_r_index2', 'b_r_index1', 'b_r_middle_null', 'b_r_middle3', 'b_r_middle2', 'b_r_middle1', 'b_r_ring_null', 'b_r_ring3', 'b_r_ring2', 'b_r_ring1', 'b_r_pinky_null', 'b_r_pinky3', 'b_r_pinky2', 'b_r_pinky1', 'b_r_wrist']

        # load 3D world coordinates of joints
        joint_world = np.ones((len(skeleton),3),dtype=np.float32)
        joint_valid = np.zeros((len(skeleton),1),dtype=np.float32)
        with open(joint_path) as f:
            for line in f:
                parsed_line = line.split()
                parsed_line = [float(x) for x in parsed_line]
                joint_idx, x_world, y_world, z_world, score_sum, num_view = parsed_line
                joint_idx = int(joint_idx) # joint_idx of the file

                if hand_type == 'right' and joint_idx > 20: # 00: right hand, 21~41: left hand
                    continue
                if hand_type == 'left' and joint_idx < 21: # 01: left hand, 0~20: right hand
                    continue
     
                joint_name = db_joint_name[joint_idx]
                joint_idx = [i for i,_ in enumerate(skeleton) if _['name'] == joint_name][0] # joint_idx which follows my 'skeleton'
               
                joint_world[joint_idx] = np.array([x_world, y_world, z_world], dtype=np.float32)
                joint_valid[joint_idx] = 1

        return joint_world, joint_valid


