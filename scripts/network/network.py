import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(nn.ReLU()(self.w_1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.layernorm1(x)
        x = x + self.dropout1(self.self_attn(x2, x2, x2, attn_mask=mask)[0])
        x2 = self.layernorm2(x)
        x = x + self.dropout2(self.ffn(x2))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class GNSSClassifier(nn.Module):
    def __init__(self, d_model=16, nhead=4, d_ff=64, num_layers=2, num_satellites=60, num_classes=21):
        super(GNSSClassifier, self).__init__()
        self.input_proj = nn.Linear(16, d_model)
        self.transformer_encoder = TransformerEncoder(d_model, nhead, d_ff, num_layers)
        self.classifier_x = nn.Linear(d_model * num_satellites, num_classes)
        self.classifier_y = nn.Linear(d_model * num_satellites, num_classes)
        self.classifier_z = nn.Linear(d_model * num_satellites, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_proj(x)
        x = x.transpose(0, 1)  # (num_satellites, batch_size, d_model)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1).contiguous().view(batch_size, -1)  # (batch_size, num_satellites * d_model)
        x_out = self.classifier_x(x)
        y_out = self.classifier_y(x)
        z_out = self.classifier_z(x)
        return F.log_softmax(x_out, dim=1), F.log_softmax(y_out, dim=1), F.log_softmax(z_out, dim=1)

# 试一下
# model = GNSSClassifier(d_model=16, nhead=4, d_ff=64, num_layers=2, num_satellites=60, num_classes=21)
# input_features = torch.randn(4, 60, 16)  # (batch_size, num_satellites, feature_dim)
# x_out, y_out, z_out = model(input_features)
# print(x_out.shape, y_out.shape, z_out.shape)  # torch.Size([4, 21]), torch.Size([4, 21]), torch.Size([4, 21])