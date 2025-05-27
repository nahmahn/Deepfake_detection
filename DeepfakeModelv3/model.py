import torch
import torch.nn as nn
from torchvision import models

class DeepfakeModel(nn.Module):
    def __init__(self, num_classes=1, lstm_hidden_size=512, lstm_layers=1, dropout_p=0.3):
        super(DeepfakeModel, self).__init__()

        base_model = models.resnext50_32x4d(pretrained=True)
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])  # Remove final pooling & FC

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)  
        feats = self.backbone(x)              
        feats = self.pool(feats).squeeze(-1).squeeze(-1)  
        feats = feats.view(B, T, -1)           

        lstm_out, _ = self.lstm(feats)         
        last_out = lstm_out[:, -1, :]         
        out = self.classifier(last_out)       

        return out.squeeze(1) 
