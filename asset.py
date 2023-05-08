import os
import numpy as np

if os.path.exists('data/scannet'):
    scannet_train_scans_320 = np.loadtxt('configs/scannetv2_train_split.txt',dtype=str).tolist()
    scannet_test_scans_320 = np.loadtxt('configs/scannetv2_test_split.txt',dtype=str).tolist()
    scannet_single = ['scannet/scene0376_02/black_320']
