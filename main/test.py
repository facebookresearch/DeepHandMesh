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
from config import cfg
import torch
from base import Tester
import torch.backends.cudnn as cudnn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--subject', type=str, dest='subject')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    assert args.test_epoch, 'Test epoch is required.'
    assert args.subject == '4', 'Testing only supports subject_4'
    return args

def main():

    args = parse_args()
    cfg.set_args(args.subject, args.gpu_ids)
    cudnn.benchmark = True

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()
    
    with torch.no_grad():
        for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):

            # forward
            out = tester.model(inputs, targets, meta_info, 'test')
            
            vis = True
            if vis:
                filename = str(itr)
                for bid in range(len(out['mesh_out_refined'])):
                    img = inputs['img'][bid].detach().cpu().numpy().transpose(1,2,0)[:,:,::-1]*255
                    cv2.imwrite(filename + '.jpg', img)
                    tester.mesh.save_obj(out['mesh_out_refined'][bid].cpu().numpy(), None, filename + '_refined.obj')
        
if __name__ == "__main__":
    main()
