from dataclasses import dataclass


@dataclass
class AppearanceModelOptimizationParams:
    lr: float = 1e-3
    eps: float = 1e-15
    gamma: float = 1
    max_steps: int = 30_000


@dataclass
class AppearanceModelParams:
    optimization: AppearanceModelOptimizationParams

    n_grayscale_factors: int = 3
    n_gammas: int = 3
    n_neurons: int = 32
    n_hidden_layers: int = 2
    n_frequencies: int = 4
    grayscale_factors_activation: str = "Sigmoid"
    gamma_activation: str = "Softplus"
