import torch.nn as nn
import torch.fft

def create_filter_groups(model, group_size=8):
    """
    Create groups of filters from all convolutional layers in ResNet.
    Returns a mapping: {(layer_name, filter_idx): group_id}
    """
    group_map = {}
    group_id = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            num_filters = module.out_channels
            for i in range(0, num_filters, group_size):
                for j in range(i, min(i + group_size, num_filters)):
                    group_map[(name, j)] = group_id
                group_id += 1
    print(f"[DEBUG] Total groups created: {len(set(group_map.values()))}")
    return group_map


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

def compute_fis_scores(filter_flux, filter_utilities, alpha=0.5, epsilon=1e-6):
    """
    Computes Filter Importance Score (FIS) for each filter.

    Args:
        filter_flux: dict {(layer_name, filter_idx): flux_value}
        filter_utilities: dict {(layer_name, filter_idx): utility_value}
        alpha: weighting exponent for chaos penalty (0 = ignore flux, 1 = full penalty)
        epsilon: small value to avoid div-by-zero

    Returns:
        fis_scores: dict {(layer_name, filter_idx): importance_score}
    """
    fis_scores = {}
    for key in filter_utilities:
        utility = filter_utilities.get(key, 0.0)
        flux = filter_flux.get(key, 1e-6)
        # Chaos-tolerant FIS: prioritize useful filters, don't over-penalize chaotic ones
        score = utility * (1.0 / (flux + epsilon)) ** alpha
        fis_scores[key] = score
    return fis_scores
