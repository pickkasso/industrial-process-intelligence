"""Automotive industry profile with keyword scoring and schema hints."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar

from process_intelligence.core.enums import ColumnRole
from process_intelligence.industries._keyword_profile import _KeywordIndustryProfile


class AutomotiveIndustryProfile(_KeywordIndustryProfile):
    """Industry profile for automotive process and vehicle datasets."""

    industry_name: ClassVar[str] = "automotive"

    keywords: ClassVar[tuple[str, ...]] = (
        "automotive",
        "vehicle_id",
        "vin",
        "engine_rpm",
        "throttle_position",
        "brake_pressure",
        "steering_angle",
        "wheel_speed",
        "fuel_rate",
        "exhaust",
        "transmission_gear",
        "odometer",
        "vehicle_speed",
        "자동차",
        "차량",
        "엔진회전수",
        "브레이크압력",
        "조향각",
        "차속",
        "배기가스",
    )

    synonyms: ClassVar[Mapping[str, tuple[str, ...]]] = MappingProxyType(
        {
            "vehicle_id": ("vehicle", "vehicleid"),
            "vin": ("vehicle_identification_number",),
            "trip_id": ("trip", "tripid"),
            "timestamp": ("time", "datetime", "event_time"),
            "vehicle_speed": ("speed", "speed_kmh"),
            "engine_rpm": ("rpm", "engine_speed"),
            "throttle_position": ("throttle", "throttle_pct"),
            "brake_pressure": ("brake", "brake_bar"),
            "steering_angle": ("steering", "steer_angle"),
            "wheel_speed": ("wheel", "wheel_rpm"),
            "fuel_rate": ("fuel", "fuel_consumption"),
            "emission_rate": ("emission", "emissions"),
        }
    )

    expected_units: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "vehicle_speed": "km/h",
            "engine_rpm": "rpm",
            "steering_angle": "deg",
            "brake_pressure": "bar",
            "fuel_rate": "L/h",
            "emission_rate": "g/km",
        }
    )

    default_roles: ClassVar[Mapping[str, ColumnRole]] = MappingProxyType(
        {
            "vehicle_id": ColumnRole.IDENTIFIER,
            "vin": ColumnRole.IDENTIFIER,
            "trip_id": ColumnRole.IDENTIFIER,
            "timestamp": ColumnRole.TIME,
            "throttle_position": ColumnRole.CONTROLLABLE_PROCESS,
            "brake_command": ColumnRole.CONTROLLABLE_PROCESS,
            "engine_rpm": ColumnRole.STATE_SENSOR,
            "vehicle_speed": ColumnRole.STATE_SENSOR,
            "steering_angle": ColumnRole.STATE_SENSOR,
            "brake_pressure": ColumnRole.STATE_SENSOR,
            "wheel_speed": ColumnRole.STATE_SENSOR,
            "road_type": ColumnRole.CONTEXT,
            "weather": ColumnRole.CONTEXT,
            "driver_id": ColumnRole.CONTEXT,
            "fuel_efficiency": ColumnRole.TARGET_QUALITY,
            "emission_rate": ColumnRole.TARGET_QUALITY,
            "failure_flag": ColumnRole.TARGET_QUALITY,
        }
    )

    physical_ranges: ClassVar[Mapping[str, tuple[float | None, float | None]]] = (
        MappingProxyType({})
    )
