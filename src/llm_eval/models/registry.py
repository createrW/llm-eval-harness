"""Provider registry: provider string -> factory + static capabilities (design §5, §6).

Each factory imports its heavy module inside its body, so importing this module never imports
``torch`` or ``transformers`` (design §5).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

from llm_eval.models.base_model import Model

if TYPE_CHECKING:
    from llm_eval.config import ModelConfig


class ModelFactory(Protocol):
    """``factory(cfg, *, seed)``: build ``cfg``'s model; a non-``None`` ``seed`` seeds the
    provider's RNG before loading (``torch.manual_seed`` for ``hf``, design §7.5)."""

    def __call__(self, cfg: ModelConfig, *, seed: int | None = None) -> Model: ...


F = TypeVar("F", bound=ModelFactory)


@dataclass(frozen=True)
class ProviderSpec:
    """How to build a provider's model, and whether it can score log-likelihoods (design §7.4)."""

    factory: ModelFactory
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


def load_model(cfg: ModelConfig, *, seed: int | None = None) -> Model:
    """Build the model of ``cfg`` with its provider's factory, seeding it with ``seed``."""
    if cfg.provider not in PROVIDERS:
        raise KeyError(f"unknown provider {cfg.provider!r}; known: {sorted(PROVIDERS)}")
    return PROVIDERS[cfg.provider].factory(cfg, seed=seed)


@register_provider("hf", supports_loglikelihood=True)
def _hf(cfg: ModelConfig, *, seed: int | None = None) -> Model:
    from llm_eval.models.hf_model import HFCausalLM

    return HFCausalLM.from_pretrained(
        cfg.path,
        name=cfg.name,
        revision=cfg.revision,
        dtype=cfg.dtype,
        device=cfg.device,
        batch_size=cfg.batch_size,
        seed=seed,
    )


@register_provider("openai", supports_loglikelihood=False)
def _openai(cfg: ModelConfig, *, seed: int | None = None) -> Model:
    from llm_eval.models.api_model import openai_model

    return openai_model(cfg, seed=seed)


@register_provider("anthropic", supports_loglikelihood=False)
def _anthropic(cfg: ModelConfig, *, seed: int | None = None) -> Model:
    from llm_eval.models.api_model import anthropic_model

    return anthropic_model(cfg, seed=seed)
