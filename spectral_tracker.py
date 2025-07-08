import torch
from utils import fft2d_filter, spectral_flux

class SpectralTracker:
    def __init__(self):
        self.previous_spectra = {}  # {layer_name: FFT_magnitude_tensor}

    def save_filter_spectra(self, model):
        """
        Stores the current magnitude spectra of all conv filters.
        Should be called at the end of each task.
        """
        self.previous_spectra.clear()

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                weights = module.weight.data.clone()
                spectrum = fft2d_filter(weights)  # shape: [C_out, C_in, H, W]
                self.previous_spectra[name] = spectrum

    def compute_flux(self, model):
        """
        Computes per-filter flux by comparing current model weights
        to the last saved spectra.

        Returns:
            filter_flux: dict { (layer_name, filter_idx): flux_value }
        """
        flux_dict = {}

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                if name not in self.previous_spectra:
                    continue  # skip if no baseline

                curr_weights = module.weight.data.clone()
                curr_spectrum = fft2d_filter(curr_weights)
                prev_spectrum = self.previous_spectra[name]

                flux_vals = spectral_flux(curr_spectrum, prev_spectrum)  # shape: [C_out]

                for i, flux in enumerate(flux_vals):
                    flux_dict[(name, i)] = flux.item()

        return flux_dict

    def compute_group_flux(self, filter_flux, group_assignments):
        """
        Averages filter flux per group based on assignments.

        Args:
            filter_flux: dict { (layer_name, filter_idx): flux_value }
            group_assignments: dict { (layer_name, filter_idx): group_id }

        Returns:
            group_flux: dict { group_id: avg_flux }
        """
        group_vals = {}
        group_counts = {}

        for key, flux in filter_flux.items():
            group_id = group_assignments.get(key, None)
            if group_id is None:
                continue
            group_vals[group_id] = group_vals.get(group_id, 0.0) + flux
            group_counts[group_id] = group_counts.get(group_id, 0) + 1

        group_flux = {gid: group_vals[gid] / group_counts[gid] for gid in group_vals}
        return group_flux

