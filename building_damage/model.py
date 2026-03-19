"""
Siamese ResNet-34 Model for Building Damage Classification.
Two branches (shared weights) process pre- and post-disaster building crops,
concatenated features are classified into 4 damage levels.
"""

import torch
import torch.nn as nn
from torchvision import models


class SiameseDamageClassifier(nn.Module):
    """
    Siamese network with shared ResNet-34 backbone for building damage
    classification from pre/post-disaster image pairs.

    Input:  pre_crop (B, 3, 128, 128), post_crop (B, 3, 128, 128)
    Output: logits (B, 4) for 4 damage classes
    """

    def __init__(self, num_classes=4, dropout=0.4, pretrained=True):
        super(SiameseDamageClassifier, self).__init__()

        # Shared ResNet-34 backbone
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        backbone = models.resnet34(weights=weights)

        # Remove the final FC layer — keep feature extractor only
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        feature_dim = 512  # ResNet-34 outputs 512-d features

        # Classifier head: takes concatenated pre+post features (512*2 = 1024)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        # Initialize classifier weights
        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, pre_crop, post_crop):
        """
        Args:
            pre_crop:  (B, 3, H, W) pre-disaster building crop
            post_crop: (B, 3, H, W) post-disaster building crop
        Returns:
            logits: (B, num_classes) classification logits
        """
        # Extract features with shared backbone
        pre_features = self.feature_extractor(pre_crop)   # (B, 512, 1, 1)
        post_features = self.feature_extractor(post_crop)  # (B, 512, 1, 1)

        # Flatten
        pre_features = pre_features.view(pre_features.size(0), -1)   # (B, 512)
        post_features = post_features.view(post_features.size(0), -1)  # (B, 512)

        # Concatenate pre+post
        combined = torch.cat([pre_features, post_features], dim=1)  # (B, 1024)

        # Classify
        logits = self.classifier(combined)  # (B, 4)
        return logits

    def get_features(self, pre_crop, post_crop):
        """Extract combined features without classification (useful for analysis)."""
        pre_features = self.feature_extractor(pre_crop).view(pre_crop.size(0), -1)
        post_features = self.feature_extractor(post_crop).view(post_crop.size(0), -1)
        return torch.cat([pre_features, post_features], dim=1)


def get_model(num_classes=4, dropout=0.4, pretrained=True):
    """Factory function to create the model."""
    model = SiameseDamageClassifier(
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained
    )
    return model


if __name__ == '__main__':
    # Quick test
    model = get_model()
    pre = torch.randn(2, 3, 128, 128)
    post = torch.randn(2, 3, 128, 128)
    out = model(pre, post)
    print(f"Model output shape: {out.shape}")  # Should be (2, 4)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
