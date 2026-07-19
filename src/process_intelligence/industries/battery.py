"""Battery industry profile with keyword scoring and schema hints."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar

from process_intelligence.core.enums import ColumnRole
from process_intelligence.industries._keyword_profile import _KeywordIndustryProfile


class BatteryIndustryProfile(_KeywordIndustryProfile):
    """Industry profile for battery process and cell datasets."""

    industry_name: ClassVar[str] = "battery"

    keywords: ClassVar[tuple[str, ...]] = (
        "battery",
        "battery_cell",
        "cell_id",
        "cycle_index",
        "state_of_charge",
        "state_of_health",
        "soc",
        "soh",
        "cathode",
        "anode",
        "electrolyte",
        "charge_capacity",
        "discharge_capacity",
        "coulombic_efficiency",
        "formation_cycle",
        "배터리",
        "이차전지",
        "양극",
        "음극",
        "전해질",
        "충전용량",
        "방전용량",
    )

    synonyms: ClassVar[Mapping[str, tuple[str, ...]]] = MappingProxyType(
        {
            "cell_id": ("cell", "cellid"),
            "batch_id": ("batch", "batchid", "lot_id"),
            "cycle_index": ("cycle", "cycle_number"),
            "timestamp": ("time", "datetime", "event_time"),
            "cell_voltage": ("voltage", "voltage_v"),
            "cell_temperature": ("temperature", "temp"),
            "charge_current": ("charging_current", "i_charge"),
            "discharge_current": ("discharging_current", "i_discharge"),
            "discharge_capacity": ("capacity", "capacity_ah"),
            "state_of_charge": ("soc", "soc_pct"),
            "state_of_health": ("soh", "soh_pct"),
            "internal_resistance": ("resistance", "ir"),
            "coulombic_efficiency": ("ce", "coulomb_efficiency"),
        }
    )

    expected_units: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "cell_voltage": "V",
            "charge_current": "A",
            "discharge_current": "A",
            "cell_temperature": "degC",
            "discharge_capacity": "Ah",
            "internal_resistance": "ohm",
        }
    )

    default_roles: ClassVar[Mapping[str, ColumnRole]] = MappingProxyType(
        {
            "cell_id": ColumnRole.IDENTIFIER,
            "batch_id": ColumnRole.IDENTIFIER,
            "timestamp": ColumnRole.TIME,
            "cycle_index": ColumnRole.TIME,
            "charge_current": ColumnRole.CONTROLLABLE_PROCESS,
            "discharge_current": ColumnRole.CONTROLLABLE_PROCESS,
            "cutoff_voltage": ColumnRole.CONTROLLABLE_PROCESS,
            "cell_voltage": ColumnRole.STATE_SENSOR,
            "cell_temperature": ColumnRole.STATE_SENSOR,
            "internal_resistance": ColumnRole.STATE_SENSOR,
            "state_of_charge": ColumnRole.STATE_SENSOR,
            "chemistry": ColumnRole.CONTEXT,
            "formation_channel": ColumnRole.CONTEXT,
            "discharge_capacity": ColumnRole.TARGET_QUALITY,
            "capacity_retention": ColumnRole.TARGET_QUALITY,
            "state_of_health": ColumnRole.TARGET_QUALITY,
            "coulombic_efficiency": ColumnRole.TARGET_QUALITY,
        }
    )

    physical_ranges: ClassVar[Mapping[str, tuple[float | None, float | None]]] = (
        MappingProxyType({})
    )
