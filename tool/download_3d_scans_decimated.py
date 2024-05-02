import os

root_path = './annotations/3D_scans_decimated'
os.makedirs(root_path, exist_ok=True)
os.chdir(root_path)

# download
url = 'https://github.com/facebookresearch/DeepHandMesh/releases/download/dataset/3D_scans_decimated.zip'
cmd = 'wget ' + url
os.system(cmd)
cmd = 'unzip 3D_scans_decimated.zip'
os.system(cmd)
for name in ('aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'ba', 'bb', 'bc', 'bd', 'be', 'bf', 'bg', 'bh', 'bi', 'bj', 'bk', 'bl', 'bm', 'bn', 'bo'):
    url = 'https://github.com/facebookresearch/DeepHandMesh/releases/download/dataset/3D_scans_decimated.tar.gz' + name + '.zip'
    cmd = 'wget ' + url
    os.system(cmd)

# verify downloaded files
cmd = 'python verify_download.py'
os.system(cmd)

# unzip downloaded files
cmd = 'sh unzip.sh'
os.system(cmd)

