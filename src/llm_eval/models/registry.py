"""Provider registry: provider string -> factory + static capabilities (design §5, §6).

Each factory imports its heavy module inside its body, so importing this module never imports
``torch`` or ``transformers`` (design §5).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from llm_eval.models.base_model import Model

if TYPE_CHECKING:
    from llm_eval.config import ModelConfig

F = TypeVar("F", bound=Callable[["ModelConfig"], Model])


@dataclass(frozen=True)
class ProviderSpec:
    """How to build a provider's model, and whether it can score log-likelihoods (design §7.4)."""

    factory: Callable[[ModelConfig], Model]
    supports_loglikelihood: bool


PROVIDERS: dict[str, ProviderSpec] = {}


def register_provider(name: str, *, supports_loglikelihood: bool) -> Callable[[F], F]:
    """Decorator: register ``factory`` in :data:`PROVIDERS` under ``name``."""

    def decorator(factory: F) -> F:
        PROVIDERS[name] = ProviderSpec(
            factory=factory, supports_loglikelihood=supports_loglikelihood
        )
        return factory

    return decorator


def load_model(cfg: ModelConfig) -> Model:
    """Build the model of ``cfg`` with its provider's factory."""
    if cfg.provider not in PROVIDERS:
        raise KeyError(f"unknown provider {cfg.provider!r}; known: {sorted(PROVIDERS)}")
    return PROVIDERS[cfg.provider].factory(cfg)


@register_provider("hf", supports_loglikelihood=True)
def _hf(cfg: ModelConfig) -> Model:
    from llm_eval.models.hf_model import HFCausalLM

    return HFCausalLM.from_pretrained(
        cfg.path,
        name=cfg.name,
        revision=cfg.revision,
        dtype=cfg.dtype,
        device=cfg.device,
        batch_size=cfg.batch_size,
    )
