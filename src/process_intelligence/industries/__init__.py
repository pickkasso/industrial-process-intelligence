"""Industry profile registry and built-in fallback profiles."""

from process_intelligence.industries.generic import (
    GenericIndustryProfile,
    create_default_registry,
)
from process_intelligence.industries.registry import IndustryRegistry

__all__ = [
    "GenericIndustryProfile",
    "IndustryRegistry",
    "create_default_registry",
]
