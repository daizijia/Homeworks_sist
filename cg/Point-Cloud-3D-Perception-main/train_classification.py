from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import *
from pointnet.model import *
from utils.my_plot import *
import torch.nn.functional as F
from tqdm import tqdm
import logging
from pathlib import Path
import numpy as np
import time


def log_print(logger, str):
    """
    save the info into log file as well as print in the terminal
    """
    logger.info(str)
    print(str)

# dataset_path = './dataset/shapenetcore_partanno_segmentation_benchmark_v0'
# dataset_path = './dataset/modelnet40_ply_hdf5_2048'
dataset_path = './dataset/object_training_datasets'
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
parser.add_argument('--num_points', type=int, default=2048, help='input point size')
parser.add_argument('--epoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, default=dataset_path, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='KITTI', help="dataset type shapenet|modelnet40|KITTI")
parser.add_argument('--attention_type', type=str, default='global_Mc', help="attention type Mc|Mn|Mcn|Mnc|global_Mc|global_Mn|no_attention")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--log_step_freq', type=str, default=30, help='batch size level log output')

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

try:
    checkpoint_path = Path('./checkpoint')
    checkpoint_path.mkdir(exist_ok=True)
    log_path = Path('./log')
    log_path.mkdir(exist_ok=True)
    nowtime = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) 
    loss_acc_path = Path('./log/%s'%nowtime)
    loss_acc_path.mkdir(exist_ok=True)
except OSError as e:
    raise e("can not create file floder!")

log_file = './log/%s_%s.log'%(opt.dataset_type, opt.attention_type)
print(log_file)
logging.basicConfig(level=logging.DEBUG,
                    filename=log_file,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s \
                    - %(lineno)d - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
    points_num = opt.num_points
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDatasetPly(
        root=opt.dataset,
        split='train')

    test_dataset = ModelNetDatasetPly(
        root=opt.dataset,
        split='test',
        data_augmentation=False)
    points_num = 2048
elif opt.dataset_type == 'KITTI':
    dataset = KITTIDataset(
        root=opt.dataset,
        split='train')

    test_dataset = KITTIDataset(
        root=opt.dataset,
        split='test',
        data_augmentation=False)  
    points_num = 64
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

classifier = PointNetCls(k=num_classes, num_points=points_num, attention_type=opt.attention_type, feature_transform=opt.feature_transform)
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))

try:
    best_model = str(checkpoint_path) + '/' + opt.dataset_type + '_' + opt.attention_type +'_best_model.pth'
    print(best_model)
    checkpoint = torch.load(best_model)
    start_epoch = checkpoint['epoch']
    best_val_acc = checkpoint['best_val_acc']
    classifier.load_state_dict(checkpoint['model_state_dict'])
    print('Use pretrain model')
except:
    start_epoch = 1
    best_val_acc = 0.0
    print('No existing model, starting training from scratch...')

if opt.model != '':
    checkpoint = torch.load(opt.model)
    start_epoch = checkpoint['epoch']
    best_val_acc = checkpoint['best_val_acc']
    classifier.load_state_dict(checkpoint['model_state_dict'])

# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
classifier.cuda()

num_batch = len(dataset) / opt.batch_size
loss_arr = []
acc_arr = []
val_loss_arr = []
val_acc_arr = []

for epoch in range(start_epoch, opt.epoch+1):
    # scheduler.step()
    total_loss = 0.0
    total_correct = 0.0
    total_val_loss = 0.0
    total_val_correct = 0.0  
    total_trainset = 0
    total_testset = 0
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        acc = correct.item()/float(opt.batch_size)
        total_loss += loss.item()
        total_correct += correct.item()
        total_trainset += points.size()[0]

        if i % opt.log_step_freq == 0 and i != 0:
            print('[EPOCH %d/%d: %d/%d] train loss: %f accuracy: %f' % (epoch, opt.epoch, i, num_batch, loss.item(), acc))

    with torch.no_grad():
        for i, data in enumerate(testdataloader, 0):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            val_loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            val_correct = pred_choice.eq(target.data).cpu().sum()
            val_acc = val_correct.item()/float(opt.batch_size)
            total_val_loss += val_loss.item()
            total_val_correct += val_correct.item()
            total_testset += points.size()[0]

    log_print(logger, '[EPOCH %d/%d] %s loss: %f accuracy: %f' % (epoch, opt.epoch, 'train', 
        total_loss / float(total_trainset), total_correct / float(total_trainset)))
    log_print(logger, '[EPOCH %d/%d] %s loss: %f accuracy: %f' % (epoch, opt.epoch, 'test ', 
        total_val_loss / float(total_testset), total_val_correct / float(total_testset)))

    loss_arr.append([epoch, total_loss / float(total_trainset)])
    val_loss_arr.append([epoch, total_val_loss / float(total_testset)])
    acc_arr.append([epoch, total_correct / float(total_trainset)])
    val_acc_arr.append([epoch, total_val_correct / float(total_testset)])

    if total_val_correct > best_val_acc:
        best_val_acc = total_val_correct
        savepath = str(checkpoint_path) + '/' + opt.dataset_type + '_' + opt.attention_type +'_best_model.pth'
        state = {
            'epoch': epoch+1,
            'best_val_acc': best_val_acc,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)

loss_arr = np.array(loss_arr)
val_loss_arr = np.array(val_loss_arr)
acc_arr = np.array(acc_arr)
val_acc_arr = np.array(val_acc_arr)

loss_file = './log/%s/%s_%s_loss.npy'%(nowtime, opt.dataset_type, opt.attention_type)
val_loss_file = './log/%s/%s_%s_val_loss.npy'%(nowtime, opt.dataset_type, opt.attention_type)
acc_file = './log/%s/%s_%s_acc.npy'%(nowtime, opt.dataset_type, opt.attention_type)
val_acc_file = './log/%s/%s_%s_val_acc.npy'%(nowtime, opt.dataset_type, opt.attention_type)
save_file = './log/%s/%s_%s_'%(nowtime, opt.dataset_type, opt.attention_type)
np.save(loss_file, loss_arr)
np.save(val_loss_file, val_loss_arr)
np.save(acc_file, acc_arr)
np.save(val_acc_file, val_acc_arr)
plot_loss(loss_file, val_loss_file, save_file)
plot_acc(acc_file, val_acc_file, save_file)
