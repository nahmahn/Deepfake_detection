import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig

class SpectralTransformerBranch(nn.Module):
    def __init__(self, hidden_size=128):
        super(SpectralTransformerBranch, self).__init__()


        config = DistilBertConfig(
            vocab_size=1,         
            max_position_embeddings=16,  
            n_layers=2,
            n_heads=2,
            dim=hidden_size,
            hidden_dim=hidden_size * 4,
            dropout=0.1,
            attention_dropout=0.1
        )

        self.encoder = DistilBertModel(config)

        
        self.token_embedding = nn.Linear(1, hidden_size)

        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 2)  
        )

    def forward(self, spectral_batch):
        
        x = spectral_batch.unsqueeze(-1) 
        x = self.token_embedding(x) 

        
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1) 
        x = torch.cat([cls_tokens, x], dim=1) 

        
        attention_mask = torch.ones(x.shape[:2], device=x.device)  
        encoded = self.encoder(inputs_embeds=x, attention_mask=attention_mask)

       
        cls_output = encoded.last_hidden_state[:, 0]  
        logits = self.classifier(cls_output) 
        return logits  
