from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class AttentionChannelNet(nn.Module):
    def __init__(self, attention_type) -> None:
        super().__init__()
        self.attention_type = attention_type
        self.is_global = self.attention_type.split('_')[0] == 'global'
        self.maxpooling = nn.MaxPool1d(3,1,0)
        self.avgpooling = nn.AvgPool1d(3,1,0)
        self.fc = nn.Linear(1,3)
        self.global_maxpooling = nn.MaxPool1d(1024,1,0)
        self.global_avgpooling = nn.AvgPool1d(1024,1,0)
        self.global_fc = nn.Linear(1,1024)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0,2,1)
        if self.is_global:
            x1 = self.global_maxpooling(x)
            x2 = self.global_maxpooling(x)
            x1 = self.sigmoid(self.global_fc(x1))
            x2 = self.sigmoid(self.global_fc(x2))
            x3 = x1 + x2
        else:
            x1 = self.maxpooling(x)
            x2 = self.avgpooling(x)
            x1 = self.sigmoid(self.fc(x1))
            x2 = self.sigmoid(self.fc(x2))
            x3 = x1 + x2
        x = x * x3
        x = x.permute(0,2,1)
        return x


class AttentionNumNet(nn.Module):
    def __init__(self, num_points, attention_type) -> None:
        super().__init__()
        self.maxpooling = nn.MaxPool1d(num_points,1,0)
        self.avgpooling = nn.AvgPool1d(num_points,1,0)
        self.fc1 = nn.Linear(3,1)
        self.fc2 = nn.Linear(1,3)
        self.sigmoid = nn.Sigmoid()

        self.attention_type = attention_type
        self.is_global = self.attention_type.split('_')[0] == 'global'
        self.maxpooling = nn.MaxPool1d(num_points,1,0)
        self.avgpooling = nn.AvgPool1d(num_points,1,0)
        self.fc1 = nn.Linear(3,1)
        self.fc2 = nn.Linear(1,3)
        self.global_fc1 = nn.Linear(1024,1)
        self.global_fc2 = nn.Linear(1,1024)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.maxpooling(x)
        x2 = self.avgpooling(x)
        x1 = x1.permute(0,2,1)
        x2 = x2.permute(0,2,1)
        if self.is_global:
            x1 = self.global_fc1(x1)
            x2 = self.global_fc1(x2)
            x1 = self.sigmoid(self.global_fc2(x1))
            x2 = self.sigmoid(self.global_fc2(x2))
        else:           
            x1 = self.fc1(x1)
            x2 = self.fc1(x2)
            x1 = self.sigmoid(self.fc2(x1))
            x2 = self.sigmoid(self.fc2(x2))
        x1 = x1.permute(0,2,1)
        x2 = x2.permute(0,2,1)
        x3 = x1 + x2
        x = x * x3
        return x


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x, trans, trans_feat


class PointNetCls(nn.Module):
    def __init__(self, k=2, num_points=2048, attention_type='Mc', feature_transform=False):
        super().__init__()
        self.attention_type = attention_type
        self.Mc = AttentionChannelNet(self.attention_type)
        self.Mn = AttentionNumNet(num_points, self.attention_type)
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.attention_type == 'Mc':
            x = self.Mc(x)
            x, trans, trans_feat = self.feat(x)
        elif self.attention_type == 'Mn':
            x = self.Mn(x)
            x, trans, trans_feat = self.feat(x)
        elif self.attention_type == 'Mcn':
            x = self.Mc(x)
            x = self.Mn(x)
            x, trans, trans_feat = self.feat(x)
        elif self.attention_type == 'Mnc':
            x = self.Mn(x)
            x = self.Mc(x)
            x, trans, trans_feat = self.feat(x)
        elif self.attention_type == 'global_Mc':
            x, trans, trans_feat = self.feat(x)
            x = self.Mc(x)
        elif self.attention_type == 'global_Mn':
            x, trans, trans_feat = self.feat(x)
            x = self.Mn(x)
        elif self.attention_type == 'no_attention':
            x, trans, trans_feat = self.feat(x)
        else:
            raise ValueError("Unexpected attention type!")
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)      
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2048))
    print(sim_data.size())
    Mc = AttentionChannelNet()
    out = Mc(sim_data)
    print('Mc', out.size())
    Mn = AttentionNumNet(num_points=2048)
    out = Mn(sim_data)
    print('Mn', out.size())

    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5, num_points=2048)
    out, _, _ = cls(sim_data)
    print('class', out.size())
