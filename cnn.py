import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CosineClassifier(nn.Module):
    def __init__(self, in_features, out_features, scale=10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return self.scale * torch.matmul(x_norm, w_norm.T)

class ResNet18Backbone(nn.Module):
    def __init__(self, num_classes=100, pretrained=False):
        super().__init__()
        base_model = models.resnet18(pretrained=pretrained)
        # Remove the final FC layer
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # Up to avgpool
        self.classifier = CosineClassifier(512, num_classes)  # 512 is ResNet18 feature dim

    def forward(self, x):
        x = self.features(x)  # Output: (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # (B, 512)
        x = self.classifier(x)
        return x

# Example usage:
# model = ResNet18Backbone(num_classes=100, pretrained=True)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# criterion = nn.CrossEntropyLoss()
