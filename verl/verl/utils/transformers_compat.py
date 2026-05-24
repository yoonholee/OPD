"""
Compatibility utilities for different versions of transformers library.
"""

from __future__ import annotations

import importlib.metadata
from functools import lru_cache
from typing import Optional

from packaging import version
from transformers import AutoModel

# Handle version compatibility for flash_attn_supports_top_left_mask.
# This function was added in newer versions of transformers.
try:
    from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask
except ImportError:
    # For older versions of transformers that don't have this function.
    # Default to False as a safe fallback for older versions.
    def flash_attn_supports_top_left_mask():
        """Fallback implementation for older transformers versions."""
        return False


@lru_cache
def is_transformers_version_in_range(min_version: Optional[str] = None, max_version: Optional[str] = None) -> bool:
    try:
        transformers_version_str = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as e:
        raise ModuleNotFoundError("The `transformers` package is not installed.") from e

    transformers_version = version.parse(transformers_version_str)

    lower_bound_check = True
    if min_version is not None:
        lower_bound_check = version.parse(min_version) <= transformers_version

    upper_bound_check = True
    if max_version is not None:
        upper_bound_check = transformers_version <= version.parse(max_version)

    return lower_bound_check and upper_bound_check


# Auto-model compatibility. Transformers 4.x used AutoModelForVision2Seq for
# some VLM/text-generation models. Newer releases use AutoModelForImageTextToText.
# Qwen3.5-2B is in the new path.
try:
    from transformers import AutoModelForImageTextToText as _AutoModelForImageTextToText
except ImportError:  # pragma: no cover, depends on transformers version
    _AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForVision2Seq as _AutoModelForVision2Seq
except ImportError:  # pragma: no cover, depends on transformers version
    _AutoModelForVision2Seq = None

AutoModelForImageTextToText = _AutoModelForImageTextToText
AutoModelForVision2Seq = _AutoModelForVision2Seq or _AutoModelForImageTextToText


def mapping_keys(auto_cls):
    mapping = getattr(auto_cls, "_model_mapping", None)
    if mapping is None:
        return ()
    return mapping.keys()


def conditional_generation_auto_class():
    return AutoModelForImageTextToText or AutoModelForVision2Seq or AutoModel


def auto_class_from_remote_name(name: str):
    if name == "AutoModelForCausalLM":
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM
    if name == "AutoModelForTokenClassification":
        from transformers import AutoModelForTokenClassification

        return AutoModelForTokenClassification
    if name in {"AutoModelForImageTextToText", "AutoModelForVision2Seq"}:
        return conditional_generation_auto_class()
    return AutoModel
