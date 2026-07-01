from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

from charl_tre.settings import Settings


class ParamRange[NumberT](NamedTuple):
    """Defines a hyperparameter with a name and a range."""

    min: NumberT
    max: NumberT
    step: NumberT | None | Literal["log"] = None

    def to_dict(self) -> dict:
        """Convert the ParamRange to a dictionary of properties for Optuna suggest methods."""
        return {
            "low": self.min,
            "high": self.max,
            "step": self.step if self.step != "log" else None,
            "log": self.step == "log",
        }


class ParamCategorical[T](NamedTuple):
    """Defines a hyperparameter with a name and a list of values."""

    values: list[T]


@dataclass
class HyperParams:
    """Hyperparameters for the Optuna study and model training."""

    # Optuna hyperparameter search space
    batch_size: ParamRange[int]
    """Batch size per GPU for training and testing both VAE and classifier."""
    lr: ParamRange[float]
    """Learning rate for the optimizer."""
    kl_final: ParamRange[float]
    """Maximum weight for the KL divergence term during annealing."""
    n_components: ParamRange[int]
    """Number of components in the Dirichlet distribution (latent space dimensionality)."""
    prior_alpha: ParamRange[float]
    """Prior alpha for the Dirichlet distribution."""
    n_fusion_layers: ParamRange[int]
    """Number of layers in the delayed fusion classifier."""
    fusion_dropout: ParamRange[float]
    """Dropout rate to use in the delayed fusion classifier."""
    conv_layers_spec: ParamCategorical[int]
    """Convolutional layers specification for the VAE encoder."""

    @staticmethod
    def from_settings(settings: Settings) -> HyperParams:
        """Create a HyperParams instance from a Settings instance."""
        params = {}

        for field_name in HyperParams.__dataclass_fields__:
            settings_name = f"hyperparam_{field_name}"

            if hasattr(settings, settings_name):
                value = getattr(settings, settings_name)
                params[field_name] = ParamCategorical(value) if isinstance(value, list) else value
                continue

            min_name = f"{settings_name}_min"
            max_name = f"{settings_name}_max"
            step_name = f"{settings_name}_step"
            if hasattr(settings, min_name) and hasattr(settings, max_name):
                params[field_name] = ParamRange(
                    min=getattr(settings, min_name),
                    max=getattr(settings, max_name),
                    step=getattr(settings, step_name, None),
                )
                continue

            message = f"Missing hyperparameter settings for '{field_name}'"
            raise AttributeError(message)

        return HyperParams(**params)
