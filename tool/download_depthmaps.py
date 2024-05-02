import os

root_path = './annotations/depthmaps'
os.makedirs(root_path, exist_ok=True)
os.chdir(root_path)

# download
url = 'https://github.com/facebookresearch/DeepHandMesh/releases/download/dataset/depthmaps.zip'
cmd = 'wget ' + url
os.system(cmd)
cmd = 'unzip depthmaps.zip'
os.system(cmd)
for name in ('aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au', 'av', 'aw', 'ax', 'ay', 'az'):
    url = 'https://github.com/facebookresearch/DeepHandMesh/releases/download/dataset/depthmaps.tar.gz' + name + '.zip'
    cmd = 'wget ' + url
    os.system(cmd)

# verify downloaded files
cmd = 'python verify_download.py'
os.system(cmd)

# unzip downloaded files
cmd = 'sh unzip.sh'
os.system(cmd)

