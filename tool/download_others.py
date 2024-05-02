import os

# download hand model
root_path = './hand_model'
os.makedirs(root_path, exist_ok=True)
os.chdir(root_path)

url = 'https://github.com/facebookresearch/DeepHandMesh/releases/download/dataset/hand_model.zip'
cmd = 'wget ' + url
os.system(cmd)
cmd = 'unzip hand_model.zip'
os.system(cmd)

os.chdir('..')
root_path = './annotations'
os.makedirs(root_path, exist_ok=True)
os.chdir(root_path)

# download keypoints
url = 'https://github.com/facebookresearch/DeepHandMesh/releases/download/dataset/keypoints.zip'
cmd = 'wget ' + url
os.system(cmd)
cmd = 'unzip keypoints.zip'
os.system(cmd)

# download keypoints
url = 'https://github.com/facebookresearch/DeepHandMesh/releases/download/dataset/KRT_512.zip'
cmd = 'wget ' + url
os.system(cmd)
cmd = 'unzip KRT_512.zip'
os.system(cmd)
