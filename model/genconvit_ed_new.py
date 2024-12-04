import torch
import torch.nn as nn
from torchvision import transforms
import timm
from .model_embedder import HybridEmbed
import torch.nn.functional as F

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class EfficientEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                # Enhanced MBConv block with skip connection
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),  # Projection
                nn.BatchNorm2d(out_c),
                Mish(),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, groups=out_c, bias=False),  # Depthwise
                nn.BatchNorm2d(out_c),
                Mish(),
                nn.Conv2d(out_c, out_c, kernel_size=1, bias=False),  # Projection
                nn.BatchNorm2d(out_c)
            )

        # Restored channel progression with residual connections
        self.conv1 = conv_block(3, 16)
        self.conv2 = conv_block(16, 32)
        self.conv3 = conv_block(32, 64)
        self.conv4 = conv_block(64, 128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        # Remove interpolation since it's unnecessary with scale_factor=1
        # and adjust the residual connections to match channel dimensions
        x1 = self.pool(self.conv1(x))
        x2 = self.pool(self.conv2(x1))  # Removed mismatched addition
        x3 = self.pool(self.conv3(x2))  # Removed mismatched addition
        x4 = self.pool(self.conv4(x3))  # Removed mismatched addition
        return x4

class EfficientDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        def upconv_block(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                Mish(),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),  # Additional conv for better features
                nn.BatchNorm2d(out_c),
                Mish()
            )

        # Restored channel progression
        self.up1 = upconv_block(128, 64)
        self.up2 = upconv_block(64, 32)
        self.up3 = upconv_block(32, 16)
        self.up4 = upconv_block(16, 8)
        self.final = nn.Sequential(
            nn.Conv2d(8, 3, kernel_size=3, padding=1),
            nn.Tanh()  # Added Tanh for better stability
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.final(x)

class GenConViTED(nn.Module):
    def __init__(self, config, pretrained=True):
        super().__init__()
        
        self.encoder = EfficientEncoder()
        self.decoder = EfficientDecoder()
        
        # Initialize backbone (ConvNeXt Tiny)
        self.backbone = timm.create_model(
            config['model']['backbone'],
            pretrained=pretrained,
            num_classes=0
        )
        
        # Initialize embedder (Swin Tiny)
        self.embedder = timm.create_model(
            config['model']['embedder'],
            pretrained=pretrained,
            num_classes=0
        )
        
        embed_dim = 384  # Restored original embedding dimension
        self.backbone.patch_embed = HybridEmbed(
            self.embedder,
            img_size=config['img_size'],
            embed_dim=embed_dim
        )
        
        # Get feature dimension from backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, config['img_size'], config['img_size'])
            feat_dim = self.backbone(dummy_input).shape[1] * 2
        
        # Enhanced classification head
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LayerNorm(feat_dim // 2),
            Mish(),
            nn.Dropout(0.2),
            nn.Linear(feat_dim // 2, feat_dim // 4),
            nn.LayerNorm(feat_dim // 4),
            Mish(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 4, config['num_classes'])
        )
        
    def forward(self, images):
        # Encode-Decode path with gradient
        encimg = self.encoder(images)
        decimg = self.decoder(encimg)
        
        # Feature extraction path
        orig_features = self.backbone(images)
        recon_features = self.backbone(decimg)
        
        # Feature fusion and classification
        x = torch.cat((orig_features, recon_features), dim=1)
        x = self.head(x)
        
        return x