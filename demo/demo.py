# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from tqdm import tqdm
import numpy as np
import cv2
import math
import torch
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
from torch.nn.parallel.data_parallel import DataParallel

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from utils import mesh
from model import get_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    assert args.test_epoch, 'Test epoch is required.'
    return args

# argument parsing
args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

# hand model settings
joint_num = 22
root_joint_idx = 21
align_joint_idx = [8, 12, 16, 20, 21]
non_rigid_joint_idx = [3,21]
hand_model_path = '../data/hand_model'
mesh = mesh.Mesh(osp.join(hand_model_path, 'hand.obj')) 
mesh.load_skeleton(osp.join(hand_model_path, 'skeleton.txt'), joint_num) # joint set is defined in here
mesh.load_skinning_weight(osp.join(hand_model_path, 'skinning_weight.txt')) 
mesh.load_local_pose(osp.join(hand_model_path, 'local_pose.txt')) 
mesh.load_global_pose(osp.join(hand_model_path, 'global_pose.txt')) 
mesh.load_global_pose_inv(osp.join(hand_model_path, 'global_pose_inv.txt')) 

# load pre-trained DeepHandMesh
model_path = './snapshot_' + args.test_epoch + '.pth.tar'
assert os.path.exists(model_path), 'Cannot find model at ' + model_path
model = get_model('test', mesh, root_joint_idx, align_joint_idx, non_rigid_joint_idx)
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# forward and save output
joint_euler = torch.zeros((1,joint_num,3)).float().cuda()
joint_euler[0,8,2] = math.pi/2
joint_euler[0,12,2] = math.pi/2
joint_euler[0,16,2] = math.pi/2
joint_euler[0,20,2] = math.pi/2
with torch.no_grad():
    out = model.module.decode(joint_euler)
mesh.save_obj(out['mesh_out_refined'][0].cpu().numpy(), None, 'mesh.obj')

