import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        if len(x.shape) == 3:
            batchsize = x.size()[0]
            x =self.relu1(self.bn1(self.conv1(x)))
            x =self.relu2(self.bn2(self.conv2(x)))
            x =self.relu3(self.bn3(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)

            x =self.relu4(self.bn4(self.fc1(x)))
            x =self.relu5(self.bn5(self.fc2(x)))
            x = self.fc3(x)

            iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
                batchsize, 1)
            if x.is_cuda:
                iden = iden.cuda()
            x = x + iden
            x = x.view(-1, 3, 3)
        else:
            step = x.size()[0]
            batchsize = x.size()[1]
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = torch.max(x, 3, keepdim=True)[0]
            x = x.view(step, batchsize, 1024)

            x = self.relu4(self.bn4(self.fc1(x)))
            x = self.relu5(self.bn5(self.fc2(x)))
            x = self.fc3(x)

            iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
                batchsize, 1)
            iden = iden.repeat(step, 1, 1)
            if x.is_cuda:
                iden = iden.cuda()
            x = x + iden
            x = x.view(step, batchsize, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        if len(x.shape) == 3:
            batchsize = x.size()[0]
            x =self.relu1(self.bn1(self.conv1(x)))
            x =self.relu2(self.bn2(self.conv2(x)))
            x =self.relu3(self.bn3(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)

            x =self.relu4(self.bn4(self.fc1(x)))
            x =self.relu5(self.bn5(self.fc2(x)))
            x = self.fc3(x)

            iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
                batchsize, 1)
            if x.is_cuda:
                iden = iden.cuda()
            x = x + iden
            x = x.view(-1, self.k, self.k)
        else:
            step = x.size()[0]
            batchsize = x.size()[1]
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = torch.max(x, 3, keepdim=True)[0]
            x = x.view(step, batchsize, 1024)

            x = self.relu4(self.bn4(self.fc1(x)))
            x = self.relu5(self.bn5(self.fc2(x)))
            x = self.fc3(x)
            iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
                batchsize, 1)
            iden = iden.repeat(step, 1, 1)
            if x.is_cuda:
                iden = iden.cuda()
            x = x + iden
            x = x.view(step, batchsize, self.k, self.k)

        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        if len(x.shape) == 3:
            B, D, N = x.size()
            trans = self.stn(x)
            x = x.transpose(2, 1)
            if D > 3:
                feature = x[:, :, 3:]
                x = x[:, :, :3]
            x = torch.bmm(x, trans)
            if D > 3:
                x = torch.cat([x, feature], dim=2)
            x = x.transpose(2, 1)
            x =self.relu1(self.bn1(self.conv1(x)))

            if self.feature_transform:
                trans_feat = self.fstn(x)
                x = x.transpose(2, 1)
                x = torch.bmm(x, trans_feat)
                x = x.transpose(2, 1)
            else:
                trans_feat = None

            pointfeat = x
            x =self.relu2(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
            if self.global_feat:
                return x, trans, trans_feat
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, N)
                return torch.cat([x, pointfeat], 1), trans, trans_feat
        else:
            T, B, D, N = x.size()
            trans = self.stn(x)
            x = x.transpose(3, 2)
            if D > 3:
                feature = x[:, :, :, 3:]
            x = x[:, :, :, :3]
            x = torch.bmm(x.view(-1, x.size(2), x.size(3)), trans.view(-1, trans.size(2), trans.size(3)))
            x =x.view( T, B, x.size(1), x.size(2))
            if D > 3:
                x = torch.cat([x, feature], dim=3)
            x = x.transpose(3, 2)

            x = self.relu1(self.bn1(self.conv1(x)))

            if self.feature_transform:
                trans_feat = self.fstn(x)
                x = x.transpose(3, 2)
                x = torch.bmm(x.view(-1, x.size(2), x.size(3)), trans_feat.view(-1, trans_feat.size(2), trans_feat.size(3)))
                x =x.view(T, B , x.size(1), x.size(2))
                x = x.transpose(3, 2)

            else:
                trans_feat = None

            pointfeat = x
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
            x = torch.max(x, 3, keepdim=True)[0]
            x = x.view(x.size(0), x.size(1), 1024)

            if self.global_feat:
                return x, trans, trans_feat
            else:
                x = x.view(x.size(0), -1, 1024, 1).repeat(1, 1, 1, N)
                return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss