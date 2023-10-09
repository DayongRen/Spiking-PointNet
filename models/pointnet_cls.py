import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
from models.spike_model import SpikeModel


class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        if len(x.shape) == 2:
            x = F.log_softmax(x, dim=1)
        else:
            x = F.log_softmax(x, dim=2)
        
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

if __name__ == '__main__':
    net = get_model()
    spike_net = SpikeModel(net)
    spike_net.eval()
    print(spike_net)
    x = torch.randn(1, 6, 4096)
    y1 = spike_net(x)
    
