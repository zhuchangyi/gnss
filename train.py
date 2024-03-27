import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from scripts.data_process.dataloader import GNSSDataset
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from datetime import datetime
from scripts.network.network import GNSSClassifier
from torch.utils.tensorboard import SummaryWriter

current_script_path = Path(__file__).resolve()
root_path = current_script_path.parents[2]
rawdata_path = root_path / "data" / "raw" / "sdc2023" / "train"
processed_path = root_path / "data" / "processed_data"
log_path = root_path / "logs" / "train_log"

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = log_path / f'gnss_experiment_{current_time}'
checkpoint_path = root_path / "checkpoints" / f'model_{current_time}'
writer = SummaryWriter(str(log_dir))


class Config:
    d_model = 16
    nhead = 4
    d_ff = 64
    num_layers = 2
    num_satellites = 60
    num_classes = 21
    batch_size = 32
    learning_rate = 1e-3
    weight_decay = 1e-5
    num_epochs = 1000


config = Config()


class GNSSLoss(nn.Module):
    def __init__(self, alpha=0.5, num_classes=21, error_range=(-10, 10)):
        super(GNSSLoss, self).__init__()
        self.alpha = alpha  # 用于平衡CELoss和MSELoss的权重
        self.num_classes = num_classes
        self.error_range = error_range
        self.error_values = torch.linspace(error_range[0], error_range[1], num_classes)  # -10到+10之间的21个值

    def forward(self, x_out, y_out, z_out, labels):
        # 计算交叉熵损失
        celoss_x = F.cross_entropy(x_out, labels[:, :, 0].argmax(dim=1))
        celoss_y = F.cross_entropy(y_out, labels[:, :, 1].argmax(dim=1))
        celoss_z = F.cross_entropy(z_out, labels[:, :, 2].argmax(dim=1))
        celoss = celoss_x + celoss_y + celoss_z

        # 计算MSELoss
        mse_x = self.calculate_mse(x_out, labels[:, :, 0])
        mse_y = self.calculate_mse(y_out, labels[:, :, 1])
        mse_z = self.calculate_mse(z_out, labels[:, :, 2])
        mseloss = mse_x + mse_y + mse_z

        # 综合两种损失
        loss = self.alpha * celoss + (1 - self.alpha) * mseloss
        return loss, celoss, mseloss

    def calculate_mse(self, outputs, labels):
        # 转换softmax输出为加权平均误差值
        predictions = torch.sum(F.softmax(outputs, dim=1) * self.error_values.to(outputs.device), dim=1)
        # 从标签的一热编码中获取实际误差值
        actual_errors = torch.sum(labels * self.error_values.to(labels.device), dim=1)
        # 计算均方误差
        mse = F.mse_loss(predictions, actual_errors)
        return mse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_dir = '/Users/park/PycharmProjects/gnss/data/processed_data/2020-06-25-00-34-us-ca-mtv-sb-101'
dataset = GNSSDataset(root_dir)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

model = GNSSClassifier(
    d_model=config.d_model,
    nhead=config.nhead,
    d_ff=config.d_ff,
    num_layers=config.num_layers,
    num_satellites=config.num_satellites,
    num_classes=config.num_classes
)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
model = model.to(device)
criterion = GNSSLoss(alpha=0.8).to(device)


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_celoss, total_mseloss = 0, 0, 0
    for batch in tqdm(loader, desc='Training'):
        optimizer.zero_grad()
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        x_out, y_out, z_out = model(inputs)
        loss, celoss, mseloss = criterion(x_out, y_out, z_out, labels)  # 修改这里
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_celoss += celoss.item()  # 新增
        total_mseloss += mseloss.item()  # 新增
    # 可以记录或返回CE和MSE的平均损失
    return total_loss / len(loader), total_celoss / len(loader), total_mseloss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_celoss, total_mseloss = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validation'):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            x_out, y_out, z_out = model(inputs)
            loss, celoss, mseloss = criterion(x_out, y_out, z_out, labels)  # 修改这里

            total_loss += loss.item()
            total_celoss += celoss.item()  # 新增
            total_mseloss += mseloss.item()  # 新增
    # 可以记录或返回CE和MSE的平均损失
    return total_loss / len(loader), total_celoss / len(loader), total_mseloss / len(loader)


for epoch in range(config.num_epochs):
    train_loss, train_celoss, train_mseloss = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_celoss, val_mseloss = validate(model, val_loader, criterion, device)

    # Log the losses
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Validation', val_loss, epoch)
    writer.add_scalar('CELoss/Train', train_celoss, epoch)
    writer.add_scalar('CELoss/Validation', val_celoss, epoch)
    writer.add_scalar('MSELoss/Train', train_mseloss, epoch)
    writer.add_scalar('MSELoss/Validation', val_mseloss, epoch)

    print(f'Epoch {epoch + 1}/{config.num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

torch.save(model.state_dict(), checkpoint_path)

# 记录超参数
writer.add_hparams(vars(config), {'HParam/Loss': val_loss})
writer.close()
