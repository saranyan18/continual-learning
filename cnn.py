import torch
import torch.nn as nn
import torch.nn.functional as F

class cnn(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(cnn, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Assuming input size is 32x32
        self.fc2 = nn.Linear(128, num_classes)

        self.conv_layers = [self.conv1, self.conv2]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_conv_weights(self):
        """
        Returns a dict of all conv filter weight tensors for spectral analysis.
        """
        weights = {}
        for i, layer in enumerate(self.conv_layers):
            weights[f'conv{i+1}'] = layer.weight.data.clone()
        return weights

    def get_intermediate_features(self, x):
        """
        Forward pass that also returns intermediate activations.
        Useful for spectral tracking or visualizations.
        """
        features = {}
        x = F.relu(self.conv1(x))
        features['conv1'] = x.clone()
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))
        features['conv2'] = x.clone()
        x = F.max_pool2d(x, 2, 2)

        return features

    def apply_group_mask(self, group_assignments, overlay_weights):
        """
        Attaches gradient hooks per conv filter to scale them by overlay_weights.

        Args:
            group_assignments: dict of (layer_name, filter_idx) → group_id
            overlay_weights: dict of group_id → scalar multiplier
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                for i in range(module.weight.shape[0]):
                    group_id = group_assignments.get((name, i), None)
                    if group_id is not None:
                        def make_hook(gid):
                            return lambda grad: grad * overlay_weights.get(gid, 1.0)
                        module.weight[i].register_hook(make_hook(group_id))

    def __str__(self):
        total_params = sum(p.numel() for p in self.parameters())
        return f"Custom CNN with {len(self.conv_layers)} conv layers, {total_params / 1e6:.2f}M parameters"
