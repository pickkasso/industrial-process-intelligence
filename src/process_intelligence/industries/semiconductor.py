"""Semiconductor industry profile with keyword scoring and schema hints."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar

from process_intelligence.core.enums import ColumnRole
from process_intelligence.industries._keyword_profile import _KeywordIndustryProfile


class SemiconductorIndustryProfile(_KeywordIndustryProfile):
    """Industry profile for semiconductor process datasets."""

    industry_name: ClassVar[str] = "semiconductor"

    keywords: ClassVar[tuple[str, ...]] = (
        "semiconductor",
        "wafer",
        "wafer_id",
        "lot_id",
        "die_id",
        "lithography",
        "photoresist",
        "etch",
        "deposition",
        "chamber_id",
        "overlay",
        "critical_dimension",
        "film_thickness",
        "sheet_resistance",
        "fab",
        "반도체",
        "웨이퍼",
        "식각",
        "증착",
        "노광",
    )

    synonyms: ClassVar[Mapping[str, tuple[str, ...]]] = MappingProxyType(
        {
            "wafer_id": ("wafer", "waferid"),
            "lot_id": ("lot", "lotid", "batch_id"),
            "die_id": ("die", "dieid"),
            "timestamp": ("time", "datetime", "event_time"),
            "chamber_id": ("chamber", "chamberid"),
            "tool_id": ("tool", "toolid", "equipment_id"),
            "recipe_id": ("recipe", "recipeid"),
            "critical_dimension": ("cd", "critical_dimension_nm"),
            "film_thickness": ("thickness", "thickness_nm"),
            "yield": ("process_yield", "yield_rate"),
            "defect_rate": ("defect", "defect_ratio"),
        }
    )

    expected_units: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "critical_dimension": "nm",
            "film_thickness": "nm",
            "chamber_pressure": "Pa",
            "rf_power": "W",
        }
    )

    default_roles: ClassVar[Mapping[str, ColumnRole]] = MappingProxyType(
        {
            "wafer_id": ColumnRole.IDENTIFIER,
            "lot_id": ColumnRole.IDENTIFIER,
            "die_id": ColumnRole.IDENTIFIER,
            "timestamp": ColumnRole.TIME,
            "etch_time": ColumnRole.CONTROLLABLE_PROCESS,
            "deposition_time": ColumnRole.CONTROLLABLE_PROCESS,
            "gas_flow": ColumnRole.CONTROLLABLE_PROCESS,
            "rf_power": ColumnRole.CONTROLLABLE_PROCESS,
            "chamber_pressure": ColumnRole.STATE_SENSOR,
            "chamber_temperature": ColumnRole.STATE_SENSOR,
            "endpoint_signal": ColumnRole.STATE_SENSOR,
            "chamber_id": ColumnRole.CONTEXT,
            "tool_id": ColumnRole.CONTEXT,
            "recipe_id": ColumnRole.CONTEXT,
            "yield": ColumnRole.TARGET_QUALITY,
            "defect_rate": ColumnRole.TARGET_QUALITY,
            "critical_dimension": ColumnRole.TARGET_QUALITY,
            "film_thickness": ColumnRole.TARGET_QUALITY,
        }
    )

    physical_ranges: ClassVar[Mapping[str, tuple[float | None, float | None]]] = (
        MappingProxyType({})
    )
