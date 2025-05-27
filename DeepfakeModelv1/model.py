import torch
import torch.nn as nn
import timm
from spectral_transformer import SpectralTransformerBranch

class SVFTSwinFusion(nn.Module):
    def __init__(self, swin_model_name='swin_tiny_patch4_window7_224', alpha=0.7):
        super(SVFTSwinFusion, self).__init__()
        self.alpha = alpha
        self.beta = 1 - alpha
        self.swin = timm.create_model(swin_model_name, pretrained=True)
        in_features = self.swin.head.in_features
        self.swin.head = nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)  
        self.swin_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 2)
        )
        self.spectral_branch = SpectralTransformerBranch(hidden_size=128)

    def forward(self, image, spectral):
        swin_feat = self.swin(image)
        if swin_feat.dim() == 4 and swin_feat.shape[1] == 7 and swin_feat.shape[2] == 7:
            swin_feat = swin_feat.permute(0, 3, 1, 2)
        swin_feat = self.global_pool(swin_feat)
        swin_feat = swin_feat.view(swin_feat.size(0), -1)
        swin_logits = self.swin_classifier(swin_feat)
        spectral_logits = self.spectral_branch(spectral)
        final_logits = self.alpha * swin_logits + self.beta * spectral_logits
        return final_logits
