import torch.nn as nn
import torch

class OverlayModulator:
    def __init__(self, model, group_assignments):
        """
        Args:
            model: your CNN model
            group_assignments: dict mapping (layer_name, filter_idx) → group_id
        """
        self.model = model
        self.group_assignments = group_assignments
        self.overlay_weights = {}  # group_id → scalar multiplier
        self.hooks = []  # Track hooks for cleanup

    def compute_overlay_weights(self, group_flux, group_utility, alpha=1.0, beta=1.0):
        """
        Combines flux + utility to compute overlay weight per group.
        Lower = preserve, higher = allow learning.

        overlay_weight[g] = sigmoid(α * flux[g] - β * utility[g])
        """
        self.overlay_weights.clear()

        for gid in group_flux:
            flux = group_flux.get(gid, 0)
            utility = group_utility.get(gid, 0)

            raw = alpha * flux - beta * utility
            overlay = 1 / (1 + torch.exp(-torch.tensor(raw)))
            self.overlay_weights[gid] = overlay.item()

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def apply_overlay(self):
        """
        Attaches hooks to all conv filters to scale their gradients using overlay_weights.
        """
        # Remove existing hooks first
        self.remove_hooks()

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                for i in range(module.weight.shape[0]):
                    gid = self.group_assignments.get((name, i), None)
                    if gid is None:
                        continue

                    weight = self.overlay_weights.get(gid, 1.0)

                    def make_hook(w):
                        return lambda grad: grad * w if grad is not None else grad

                    hook = module.weight[i].register_hook(make_hook(weight))
                    self.hooks.append(hook)
