# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

root_path = './images'
os.makedirs(root_path, exist_ok=True)
os.chdir(root_path)

# download
url = 'https://github.com/facebookresearch/DeepHandMesh/releases/download/dataset/images.zip'
cmd = 'wget ' + url
os.system(cmd)
cmd = 'unzip images.zip'
os.system(cmd)
for name in ('aa', 'ab', 'ac', 'ad', 'ae', 'af'):
    url = 'https://github.com/facebookresearch/DeepHandMesh/releases/download/dataset/images.tar.gz' + name + '.zip'
    cmd = 'wget ' + url
    os.system(cmd)

# verify downloaded files
cmd = 'python verify_download.py'
os.system(cmd)

# unzip downloaded files
cmd = 'sh unzip.sh'
os.system(cmd)

