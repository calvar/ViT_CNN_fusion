import torch
import torch.nn as nn
import torch.nn.functional as F

# Patch embedding
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # Class token
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim)) # Position embedding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x) # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, embed_dim)
        cls_token = self.cls_token.expand(B, -1, -1) # (B, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1) # (B, num_patches + 1, embed_dim)
        x = x + self.pos_embed
        return x
    
# Multi-layer Perceptron
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, drop_rate=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_dim=512, drop_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, drop_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention block. Accepts Q, K, V all as x
        attn_output = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + attn_output  # Residual connection
        # MLP block
        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output  # Residual connection
        return x
    
# Vision Transformer Model
class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=256, num_heads=8, depth=6, mlp_dim=512, drop_rate=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.encoder = nn.Sequential(
            *[TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, drop_rate) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)  # (B, num_patches + 1, embed_dim)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]  # Extract the class token
        return self.head(cls_token)  # (B, num_classes)