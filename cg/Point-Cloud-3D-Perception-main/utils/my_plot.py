import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def plot_loss(loss_file, val_loss_file, save_file):
    loss = np.load(loss_file)
    val_loss = np.load(val_loss_file)
    max_epoch = int(loss[-1, 0]) + 1
    epoch = [i for i in range(0, max_epoch+1, 20)]
    plt.figure()
    plt.plot(loss[:,0], loss[:,1], '-', color='r', label='loss')
    plt.plot(val_loss[:,0], val_loss[:,1], '-', color='b', label='val_loss')
    plt.xticks(epoch)    
    plt.legend(loc='best',frameon=False)
    plt.title('Loss Curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(save_file+'loss.png')
    plt.show()


def plot_acc(acc_file, val_acc_file, save_file):
    acc = np.load(acc_file)
    val_acc = np.load(val_acc_file)
    max_epoch = int(acc[-1, 0]) + 1
    epoch = [i for i in range(0, max_epoch+1, 20)]
    plt.figure()
    plt.plot(acc[:,0], acc[:,1], '-', color='r', label='acc')
    plt.plot(val_acc[:,0], val_acc[:,1], '-', color='b', label='val_acc')
    plt.xticks(epoch)    
    plt.legend(loc='best',frameon=False)
    plt.title('Acc Curve')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.savefig(save_file+'acc.png')
    plt.show()


def read_max_acc(root, save_path):
    for time_stamp in os.listdir(root):
        if os.path.isdir(time_stamp):
            # print(time_stamp)
            data_path = os.path.join(root, time_stamp)
            for data in os.listdir(data_path):
                content = data.split('.')[0]
                # print(content)
                params = content.split('_')
                val = params[-2]
                acc_and_loss = params[-1]
                if acc_and_loss == 'acc' and val == 'val':
                    print(content)
                    val_acc_path = os.path.join(data_path, data)
                    val_acc = np.load(val_acc_path)
                    max_val_acc = 0.0
                    for i in range(val_acc.shape[0]):
                        if (val_acc[i][1] > max_val_acc).any():
                            max_val_acc = val_acc[i][1]
                    # print(max_val_acc)
                    with open(save_path, 'a') as f:
                        f.write(content + ' ' + str(max_val_acc) + '\n')


if __name__ == '__main__':
    timestamp = '2022-12-17-14:48:40'
    # KITTI modelnet40 
    dataset = 'modelnet40'
    # Mc Mn Mcn Mnc no_attention global_Mc global_Mn
    attention = 'global_Mc'
    loss_file = './%s/%s_%s_loss.npy'%(timestamp, dataset, attention)
    val_loss_file = './%s/%s_%s_val_loss.npy'%(timestamp, dataset, attention)
    acc_file = './%s/%s_%s_acc.npy'%(timestamp, dataset, attention)
    val_acc_file = './%s/%s_%s_val_acc.npy'%(timestamp, dataset, attention)
    save_file = './%s_%s_'%(dataset, attention)
    plot_loss(loss_file, val_loss_file, save_file)
    plot_acc(acc_file, val_acc_file, save_file)

    # root = './'
    # save_path = './max_acc_list.txt'
    # read_max_acc(root, save_path)