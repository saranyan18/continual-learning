import torch.nn as nn
import torch.fft

def create_filter_groups(model, group_size=8):
    """
    Returns a dict mapping (layer_name, filter_idx) → group_id.
    """
    group_assignments = {}
    group_id = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            out_channels = module.out_channels
            for f in range(out_channels):
                group_assignments[(name, f)] = group_id
                if (f + 1) % group_size == 0:
                    group_id += 1
            if out_channels % group_size != 0:
                group_id += 1

    return group_assignments


def fft2d_filter(filter_weights):
    """
    Compute 2D FFT with proper padding for small filters
    """
    # Ensure minimum size for stable FFT
    if filter_weights.shape[-1] < 4 or filter_weights.shape[-2] < 4:
        pad_h = max(0, 4 - filter_weights.shape[-2])
        pad_w = max(0, 4 - filter_weights.shape[-1])
        filter_weights = torch.nn.functional.pad(filter_weights, (0, pad_w, 0, pad_h))
    fft = torch.fft.fft2(filter_weights, norm='ortho')
    return torch.abs(fft)

def spectral_flux(current_mag, previous_mag, epsilon=1e-8):
    """
    Compute spectral flux with numerical stability
    """
    if current_mag.shape != previous_mag.shape:
        raise ValueError(f"Shape mismatch: {current_mag.shape} vs {previous_mag.shape}")
    # Normalize magnitudes to prevent scale issues
    current_norm = current_mag / (torch.norm(current_mag, dim=(-2, -1), keepdim=True) + epsilon)
    previous_norm = previous_mag / (torch.norm(previous_mag, dim=(-2, -1), keepdim=True) + epsilon)
    diff = current_norm - previous_norm
    flux = torch.norm(diff.view(diff.shape[0], -1), dim=1)
    return flux
