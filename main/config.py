# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import os.path as osp
import sys
import math
import numpy as np

class Config:
    ## input, output
    input_img_shape = (256,256)
    rendered_img_shape = (256,256)
    depth_min = 1
    depth_max = 99999
    render_view_num = 6
    backbone_img_feat_dim = 512
    id_code_dim = 32
    bone_step_size = 0.1
    
    ## model
    resnet_type = 50 # 18, 34, 50, 101, 152

    ## training config
    lr_dec_epoch = [30, 32]
    end_epoch = 35
    lr = 1e-4
    lr_dec_factor = 10
    train_batch_size = 8
    loss_penet_r_weight = 1 
    loss_penet_nr_weight = 5
    loss_lap_weight = 5

    ## testing config
    test_batch_size = 1

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')

    ## others
    num_thread = 40
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False

    def set_args(self, subject, gpu_ids, continue_train=False):
        self.subject = subject
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))
        
        self.model_dir = osp.join(self.model_dir, 'subject_' + str(self.subject))
        self.vis_dir = osp.join(self.vis_dir, 'subject_' + str(self.subject))
        self.log_dir = osp.join(self.log_dir, 'subject_' + str(self.subject))
        self.result_dir = osp.join(self.result_dir, 'subject_' + str(self.subject))

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)

