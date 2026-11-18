"""Default model id of the in-loop LLM baselines (``LLM_MODEL`` overrides it).

Standard library only and without import side effects, so the adapters (package import)
and the isolated workers (imported from this directory) share the same constant.
"""

DEFAULT_LLM_MODEL = "Qwen3.5-122B-A10B-GPTQ-Int4"

__all__ = ["DEFAULT_LLM_MODEL"]
