"""Industry profile registry and built-in fallback profiles."""

from process_intelligence.industries.automotive import AutomotiveIndustryProfile
from process_intelligence.industries.battery import BatteryIndustryProfile
from process_intelligence.industries.generic import (
    GenericIndustryProfile,
    create_default_registry,
)
from process_intelligence.industries.registry import IndustryRegistry
from process_intelligence.industries.semiconductor import SemiconductorIndustryProfile

__all__ = [
    "AutomotiveIndustryProfile",
    "BatteryIndustryProfile",
    "GenericIndustryProfile",
    "IndustryRegistry",
    "SemiconductorIndustryProfile",
    "create_default_registry",
]
