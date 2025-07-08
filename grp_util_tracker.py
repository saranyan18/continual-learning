import torch
from collections import defaultdict
import torch.nn as nn

class GroupUtilityTracker:
    def __init__(self, model, group_assignments):
        """
        Args:
            model: CNN model
            group_assignments: dict mapping (layer_name, filter_idx) → group_id
        """
        self.model = model
        self.group_assignments = group_assignments
        self.activations = {}
        self.gradients = {}
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(self._make_forward_hook(name))
                module.register_full_backward_hook(self._make_backward_hook(name))

    def _make_forward_hook(self, layer_name):
        def hook(module, input, output):
            self.activations[layer_name] = output.detach()
        return hook

    def _make_backward_hook(self, layer_name):
        def hook(module, grad_input, grad_output):
            self.gradients[layer_name] = grad_output[0].detach()
        return hook

    def compute_group_utilities(self):
        """Returns group_id → average utility with proper validation"""
        group_scores = defaultdict(list)

        for layer_name in self.activations:
            if layer_name not in self.gradients:
                continue

            acts = self.activations[layer_name]
            grads = self.gradients[layer_name]

            if acts is None or grads is None:
                continue

            # Ensure shapes match
            if acts.shape != grads.shape:
                print(f"Warning: Shape mismatch in {layer_name}: {acts.shape} vs {grads.shape}")
                continue

            B, C, H, W = acts.shape

            for f in range(C):
                key = (layer_name, f)
                gid = self.group_assignments.get(key, None)
                if gid is None:
                    continue

                # Compute utility with numerical stability
                act_grad = acts[:, f] * grads[:, f]
                score = torch.mean(torch.abs(act_grad)).item()
                if not torch.isfinite(torch.tensor(score)):
                    continue
                group_scores[gid].append(score)

        # Average per group with fallback
        group_utilities = {}
        for gid, scores in group_scores.items():
            if scores:
                group_utilities[gid] = sum(scores) / len(scores)
            else:
                group_utilities[gid] = 0.0
        return group_utilities

    def reset(self):
        self.activations.clear()
        self.gradients.clear()
