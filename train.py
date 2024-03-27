import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from scripts.data_process.dataloader import GNSSDataset
from pathlib import Path
import torch.nn.functional as F
from scripts.network.network import GNSSClassifier

batch_size = 32
num_epochs = 100
learning_rate = 1e-3
weight_decay = 1e-5


class GNSSLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super(GNSSLoss, self).__init__()
        self.alpha = alpha

    def forward(self, x_out, y_out, z_out, labels):
        batch_size, num_satellites, num_classes = labels.shape

        # 将标签展平为 (batch_size * num_satellites, num_classes)
        labels_flat = labels.view(-1, num_classes)

        # 计算交叉熵损失
        x_loss = F.cross_entropy(x_out, labels_flat[:, 0], reduction='none')
        y_loss = F.cross_entropy(y_out, labels_flat[:, 1], reduction='none')
        z_loss = F.cross_entropy(z_out, labels_flat[:, 2], reduction='none')
        ce_loss = (x_loss + y_loss + z_loss).view(batch_size, num_satellites).mean(dim=1)

        # 计算均方误差损失
        x_preds = x_out.max(dim=2)[1].float()
        y_preds = y_out.max(dim=2)[1].float()
        z_preds = z_out.max(dim=2)[1].float()
        x_targets = labels_flat[:, 0].float() - 10  # 解码距离值
        y_targets = labels_flat[:, 1].float() - 10
        z_targets = labels_flat[:, 2].float() - 10
        x_mse = (x_preds - x_targets) ** 2
        y_mse = (y_preds - y_targets) ** 2
        z_mse = (z_preds - z_targets) ** 2
        mse_loss = (x_mse + y_mse + z_mse).view(batch_size, num_satellites).mean(dim=1)

        # 结合交叉熵损失和均方误差损失
        loss = self.alpha * ce_loss + (1 - self.alpha) * mse_loss
        return loss.mean()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# root_dir = '/Users/park/PycharmProjects/gnss/data/processed_data'
# dataset = GNSSDataset(root_dir)
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#
# model = GNSSClassifier(d_model=16, nhead=4, d_ff=64, num_layers=2, num_satellites=60, num_classes=21)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# criterion = nn.CrossEntropyLoss()


