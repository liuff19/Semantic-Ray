import abc
import glob
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.io import imread
from natsort import natsorted

from utils.base_utils import downsample_gaussian_blur, pose_inverse
from .semantic_utils import PointSegClassMapping


class BaseDatabase(abc.ABC):
    def __init__(self, database_name):
        self.database_name = database_name

    @abc.abstractmethod
    def get_image(self, img_id):
        pass

    @abc.abstractmethod
    def get_K(self, img_id):
        pass

    @abc.abstractmethod
    def get_pose(self, img_id):
        pass

    @abc.abstractmethod
    def get_img_ids(self, check_depth_exist=False):
        pass

    @abc.abstractmethod
    def get_bbox(self, img_id):
        pass

    @abc.abstractmethod
    def get_depth(self, img_id):
        pass

    @abc.abstractmethod
    def get_mask(self, img_id):
        pass

    @abc.abstractmethod
    def get_depth_range(self, img_id):
        pass


class ScannetDatabase(BaseDatabase):
    def __init__(self, database_name):
        super().__init__(database_name)
        _, self.scene_name, background_size = database_name.split('/')
        background, image_size = background_size.split('_')
        image_size = int(image_size)
        self.image_size = image_size
        self.background = background
        self.root_dir = f'data/scannet/{self.scene_name}'
        self.ratio = image_size / 1296
        self.h, self.w = int(self.ratio*972), int(image_size)

        rgb_paths = [x for x in glob.glob(os.path.join(
            self.root_dir, "color", "*")) if (x.endswith(".jpg") or x.endswith(".png"))]
        rgb_paths = sorted(rgb_paths)

        K = np.loadtxt(
            f'{self.root_dir}/intrinsic/intrinsic_color.txt').reshape([4, 4])[:3, :3]
        # After resize, we need to change the intrinsic matrix
        K[:2, :] *= self.ratio
        self.K = K

        self.img_ids = []
        for i, rgb_path in enumerate(rgb_paths):
            pose = self.get_pose(i)
            if np.isinf(pose).any() or np.isnan(pose).any():
                continue
            self.img_ids.append(f'{i}')

        self.img_id2imgs = {}
        # mapping from scanntet class id to nyu40 class id
        mapping_file = 'data/scannet/scannetv2-labels.combined.tsv'
        mapping_file = pd.read_csv(mapping_file, sep='\t', header=0)
        scan_ids = mapping_file['id'].values
        nyu40_ids = mapping_file['nyu40id'].values
        scan2nyu = np.zeros(max(scan_ids) + 1, dtype=np.int32)
        for i in range(len(scan_ids)):
            scan2nyu[scan_ids[i]] = nyu40_ids[i]
        self.scan2nyu = scan2nyu
        self.label_mapping = PointSegClassMapping(
            valid_cat_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                           11, 12, 14, 16, 24, 28, 33, 34, 36, 39],
            max_cat_id=40
        )

    def get_image(self, img_id):
        if img_id in self.img_id2imgs:
            return self.img_id2imgs[img_id]
        img = imread(os.path.join(
            self.root_dir, 'color', f'{int(img_id)}.jpg'))
        if self.w != 1296:
            img = cv2.resize(downsample_gaussian_blur(
                img, self.ratio), (self.w, self.h), interpolation=cv2.INTER_LINEAR)

        return img

    def get_K(self, img_id):
        return self.K.astype(np.float32)

    def get_pose(self, img_id):
        pose = np.loadtxt(
            f'{self.root_dir}/pose/{int(img_id)}.txt').reshape([4, 4])[:3, :]
        pose = pose_inverse(pose)
        return pose.copy()

    def get_img_ids(self, check_depth_exist=False):
        return self.img_ids

    def get_bbox(self, img_id):
        raise NotImplementedError

    def get_depth(self, img_id):
        h, w, _ = self.get_image(img_id).shape
        img = Image.open(f'{self.root_dir}/depth/{int(img_id)}.png')
        depth = np.asarray(img, dtype=np.float32) / 1000.0  # mm -> m
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        return depth

    def get_mask(self, img_id):
        h, w, _ = self.get_image(img_id).shape
        return np.ones([h, w], bool)

    def get_depth_range(self, img_id):
        return np.asarray((0.1, 10.0), np.float32)

    def get_label(self, img_id):
        h, w, _ = self.get_image(img_id).shape
        img = Image.open(f'{self.root_dir}/label-filt/{int(img_id)}.png')
        label = np.asarray(img, dtype=np.int32)
        label = np.ascontiguousarray(label)
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        label = label.astype(np.int32)
        label = self.scan2nyu[label]
        return self.label_mapping(label)


class ReplicaDatabase(BaseDatabase):
    def __init__(self, database_name):
        super().__init__(database_name)
        _, self.scene_name, self.seq_id, background_size = database_name.split('/')
        background, image_size = background_size.split('_')
        self.image_size = int(image_size)
        self.background = background
        self.root_dir = f'data/replica/{self.scene_name}/{self.seq_id}'
        self.ratio = self.image_size / 640
        self.h, self.w = int(self.ratio*480), int(self.image_size)

        rgb_paths = [x for x in glob.glob(os.path.join(
            self.root_dir, "rgb", "*")) if (x.endswith(".jpg") or x.endswith(".png"))]
        self.rgb_paths = natsorted(rgb_paths)
        # DO NOT use sorted() here!!! it will sort the name in a wrong way since the name is like rgb_1.jpg

        depth_paths = [x for x in glob.glob(os.path.join(
            self.root_dir, "depth", "*")) if (x.endswith(".jpg") or x.endswith(".png"))]
        self.depth_paths = natsorted(depth_paths)

        label_paths = [x for x in glob.glob(os.path.join(
            self.root_dir, "semantic_class", "*")) if (x.endswith(".jpg") or x.endswith(".png"))]
        self.label_paths = natsorted(label_paths)

        # Replica camera intrinsics
        # Pinhole Camera Model
        fx, fy, cx, cy, s = 320.0, 320.0, 319.5, 229.5, 0.0
        if self.ratio != 1.0:
            fx, fy, cx, cy = fx * self.ratio, fy * self.ratio, cx * self.ratio, cy * self.ratio
        self.K = np.array([[fx, s, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        c2ws = np.loadtxt(f'{self.root_dir}/traj_w_c.txt',
                          delimiter=' ').reshape(-1, 4, 4).astype(np.float32)
        self.poses = []
        transf = np.diag(np.asarray([1, -1, -1]))
        num_poses = c2ws.shape[0]
        for i in range(num_poses):
            pose = c2ws[i][:3]
            # Change the pose to OpenGL coordinate system
            pose = transf @ pose
            pose = pose_inverse(pose)
            self.poses.append(pose)

        self.img_ids = []
        for i, rgb_path in enumerate(self.rgb_paths):
            self.img_ids.append(i)

        self.label_mapping = PointSegClassMapping(
            valid_cat_ids=[
                3, 7, 8, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 22, 23, 26, 29, 31,
                34, 35, 37, 40, 44, 47, 52, 54, 56,
                59, 60, 61, 62, 63, 64, 65, 70, 71,
                76, 78, 79, 80, 82, 83, 87, 88, 91,
                92, 93, 95, 97, 98
            ],
            max_cat_id=101
        )

    def get_image(self, img_id):
        img = imread(self.rgb_paths[img_id])
        if self.w != 640:
            img = cv2.resize(downsample_gaussian_blur(
                img, self.ratio), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return img

    def get_K(self, img_id):
        return self.K

    def get_pose(self, img_id):
        pose = self.poses[img_id]
        return pose.copy()

    def get_img_ids(self, check_depth_exist=False):
        return self.img_ids

    def get_bbox(self, img_id):
        raise NotImplementedError

    def get_depth(self, img_id):
        h, w, _ = self.get_image(img_id).shape
        img = Image.open(self.depth_paths[img_id])
        depth = np.asarray(img, dtype=np.float32) / 1000.0  # mm to m
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        return depth

    def get_mask(self, img_id):
        h, w, _ = self.get_image(img_id).shape
        return np.ones([h, w], bool)

    def get_depth_range(self, img_id):
        return np.asarray((0.1, 6.0), np.float32)

    def get_label(self, img_id):
        h, w, _ = self.get_image(img_id).shape
        img = Image.open(self.label_paths[img_id])
        label = np.asarray(img, dtype=np.int32)
        label = np.ascontiguousarray(label)
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        label = label.astype(np.int32)
        return self.label_mapping(label)


def parse_database_name(database_name: str) -> BaseDatabase:
    name2database = {
        'scannet': ScannetDatabase,
    }
    database_type = database_name.split('/')[0]
    if database_type in name2database:
        return name2database[database_type](database_name)
    else:
        raise NotImplementedError


def get_database_split(database: BaseDatabase, split_type='val'):
    database_name = database.database_name
    if split_type.startswith('val'):
        splits = split_type.split('_')
        depth_valid = not(len(splits) > 1 and splits[1] == 'all')
        if database_name.startswith('scannet'):
            img_ids = database.get_img_ids()
            train_ids = img_ids[:700:5]
            val_ids = img_ids[2:700:20]
            if len(val_ids) > 10:
                val_ids = val_ids[:10]
        else:
            raise NotImplementedError
    elif split_type.startswith('test'):
        splits = split_type.split('_')
        depth_valid = not(len(splits) > 1 and splits[1] == 'all')
        if database_name.startswith('scannet'):
            img_ids = database.get_img_ids()
            train_ids = img_ids[:700:5]
            val_ids = img_ids[2:700:20]
            if len(val_ids) > 10:
                val_ids = val_ids[:10]
        else:
            raise NotImplementedError
    elif split_type.startswith('video'):
        img_ids = database.get_img_ids()
        train_ids = img_ids[::2]
        val_ids = img_ids[25:-25:2]
    else:
        raise NotImplementedError
    print('train_ids:\n', train_ids)
    print('val_ids:\n', val_ids)
    return train_ids, val_ids
