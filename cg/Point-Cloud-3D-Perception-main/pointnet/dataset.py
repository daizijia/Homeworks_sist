from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
import h5py
import glob

def get_segmentation_classes(root):
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    meta = {}

    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    for item in cat:
        dir_seg = os.path.join(root, cat[item], 'points_label')
        dir_point = os.path.join(root, cat[item], 'points')
        fns = sorted(os.listdir(dir_point))
        meta[item] = []
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'w') as f:
        for item in cat:
            datapath = []
            num_seg_classes = 0
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))

            for i in tqdm(range(len(datapath))):
                l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
                if l > num_seg_classes:
                    num_seg_classes = l

            print("category {} num segmentation classes {}".format(item, num_seg_classes))
            f.write("{}\t{}\n".format(item, num_seg_classes))

def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))

class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        # print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid+'.pts'),
                                        os.path.join(self.root, category, 'points_label', uuid+'.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)

class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []

        self.cat = {}
        for category in os.listdir(root):
            category_dir = os.path.join(root, category, split)
            for i, fn in enumerate(os.listdir(category_dir)):
                full_fn = os.path.join(category_dir, fn)
                self.fns.append([category, full_fn])            
                # print('{} {}'.format(fn, full_fn))

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        print(self.cat)
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn[0]]
        # print(fn, cls)
        with open(fn[1], 'r') as f:
            f.readline()
            lines = f.readlines()
            pts = []
            for line in lines:
                points = line.split(' ')
                pts.append([points[0], points[1], points[2]])
        pts = np.array(pts, dtype=np.float32)
        # print(pts.shape)
        # pts = np.loadtxt(fn[1]).astype(np.float32)
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls


class ModelNetDatasetPly(data.Dataset):
    def __init__(self,
                 root,
                 split='train',
                 data_augmentation=True):
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.all_data = []
        self.all_label = []

        for h5_name in glob.glob(os.path.join(self.root, 'ply_data_%s*.h5'%self.split)):
            f = h5py.File(h5_name)
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            self.all_data.append(data)
            self.all_label.append(label)
        self.all_data = np.concatenate(self.all_data, axis=0)
        self.all_label = np.concatenate(self.all_label, axis=0)
        # print(self.all_data.shape)
        # print(self.all_label.shape)

        self.cat = {}
        with open(os.path.join(root, 'modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        print(self.cat)
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        point_set = self.all_data[index]
        cls = int(self.all_label[index])
        # print(point_set.shape)

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls

    def __len__(self):
        return len(self.all_data)

class KITTIDataset(data.Dataset):
    def __init__(self, root, split = 'train', data_augmentation=True) -> None:
        super().__init__()
        self.data_augmentation = data_augmentation
        self.pts_set = []
        self.label = []
        self.classes = {}

        with open(os.path.join(root, 'clsname_to_index.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                value = line.strip().split(' ')
                self.classes[value[0]] = int(value[1])
        print(self.classes)

        for category in os.listdir(root):
            category_path = os.path.join(root, category)
            if os.path.isdir(category_path):
                category_dir = os.path.join(category_path, split)
                print(category_dir)
                for item in os.listdir(category_dir):
                    data_path = os.path.join(category_dir, item)
                    self.label.append(self.classes[category])
                    points = []
                    with open(data_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            value = line.strip().split(',')
                            point = [value[i] for i in range(3)]
                            points.append(point)   
                    self.pts_set.append(points)
        self.pts_set = np.array(self.pts_set, dtype=np.float32)
        # print(self.pts_set.shape)
        # print(len(self.label))
                    
        # for category in os.listdir(root):
        #     category_path = os.path.join(root, category)
        #     if os.path.isdir(category_path):
        #         category_dir = os.path.join(category_path, split)
        #         print(category_dir)
        #         for item in os.listdir(category_dir):
        #             self.data_path.append(item)
        #             self.cls.append(self.classes[category])
        # print(len(self.data_path))
        # print(len(self.cls))


    def __getitem__(self, index):
        point_set = self.pts_set[index]
        cls = self.label[index]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls

    def __len__(self):
        return len(self.pts_set)



if __name__ == '__main__':
    # dataset = sys.argv[1]
    # datapath = sys.argv[2]

    # if dataset == 'shapenet':
    #     d = ShapeNetDataset(root = datapath, class_choice = ['Chair'])
    #     print(len(d))
    #     ps, seg = d[0]
    #     print(ps.size(), ps.type(), seg.size(),seg.type())

    #     d = ShapeNetDataset(root = datapath, classification = True)
    #     print(len(d))
    #     ps, cls = d[0]
    #     print(ps.size(), ps.type(), cls.size(),cls.type())
    #     # get_segmentation_classes(datapath)

    # if dataset == 'modelnet':
    #     # gen_modelnet_id(datapath)
    #     d = ModelNetDataset(root=datapath)
    #     print(len(d))
    #     print(d[0])

    datapath = '/home/terry/Project/Point-Cloud-3D-Perception/dataset/object_training_datasets'
    d = KITTIDataset(root=datapath, split='test')
    ps, cls = d[0]
    print(ps.size(), ps.type(), cls.size(),cls.type())

    datapath = '/home/terry/Project/Point-Cloud-3D-Perception/dataset/modelnet40_ply_hdf5_2048'
    d = ModelNetDatasetPly(root=datapath, split='test')
    ps, cls = d[0]
    print(ps.size(), ps.type(), cls.size(),cls.type())

    # datapath = '/home/terry/Project/Point-Cloud-3D-Perception/dataset/shapenetcore_partanno_segmentation_benchmark_v0'
    # d = ShapeNetDataset(root = datapath, classification = True)
    # print(len(d))
    # ps, cls = d[0]
    # print(ps.size(), ps.type(), cls.size(),cls.type())
