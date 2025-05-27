import torch
import torch.nn as nn
import timm

class DeepfakeBaseline(nn.Module):
    def __init__(self, backbone_name='efficientnet_b4', lstm_hidden_size=256, lstm_layers=1):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        
        self.lstm = nn.LSTM(
            input_size=self.backbone.num_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        
        feats = self.backbone(x)  
        feats = feats.view(B, T, -1)  
        
        lstm_out, (h_n, c_n) = self.lstm(feats)  
        
        video_feat = lstm_out[:, -1, :] 
        
        out = self.classifier(video_feat)
        
        return out.squeeze(1)
