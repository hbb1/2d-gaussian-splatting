from typing import Literal
from dataclasses import dataclass
import numpy as np


@dataclass
class TCNNEncodingConfig:
    type: Literal["frequency", "hashgrid", "densegrid", "identity", "SphericalHarmonics", "none"] = "hashgrid"
    # frequency config
    n_frequencies: int = 4
    # Spherical Harmonics config
    degree: int = 4
    # hashgrid config
    n_features_per_level: int = 4
    log2_hashmap_size: int = 19
    max_resolution: int = 2048
    # both hashgrid and densegrid config
    n_levels: int = 8
    base_resolution: int = 16
    # densegrid config
    per_level_scale: float = 1.405

    def get_encoder_config(self, n_input_channels: int):
        if self.type == "frequency":
            return {
                "n_dims_to_encode": n_input_channels,
                "otype": "Frequency",
                "n_frequencies": self.n_frequencies,
            }

        if self.type == "SphericalHarmonics":
            return {
                "n_dims_to_encode": n_input_channels,
                "otype": "SphericalHarmonics",
                "degree": self.degree,
            }

        if self.type == "hashgrid":
            max_res = self.max_resolution
            min_res = self.base_resolution
            num_levels = self.n_levels
            growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1)) if num_levels > 1 else 1
            return {
                "n_dims_to_encode": n_input_channels,
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": self.n_features_per_level,
                "log2_hashmap_size": self.log2_hashmap_size,
                "base_resolution": min_res,
                "per_level_scale": growth_factor,
            }

        if self.type == "densegrid":
            return {
                "n_dims_to_encode": n_input_channels,
                "otype": "DenseGrid",
                "n_levels": self.n_levels,
                "base_resolution": self.base_resolution,
                "per_level_scale": self.per_level_scale,
            }

        return {
            "n_dims_to_encode": n_input_channels,
            "otype": "Identity",
        }
