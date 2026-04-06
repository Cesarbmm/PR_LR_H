"""Local SQLite-backed scenario catalog for Logistics Drift worlds."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from chromahack.utils.paths import resolve_project_path


LOGISTICS_CATALOG_PATH = resolve_project_path("data/logistics_catalog.sqlite")
LOGISTICS_DISTRICT_IDS = (31, 32, 33, 34, 35, 36, 37, 38)

_DISTRICT_ROWS: tuple[dict[str, Any], ...] = (
    {
        "district_id": 31,
        "name": "Depot Loop",
        "story_phase": "logistics_birth",
        "world_split": "train",
        "primary_event_type": "route_delay",
        "time_low": 0.18,
        "time_high": 0.34,
        "step_low": 1.04,
        "step_high": 1.16,
        "hostile_low": 0,
        "hostile_high": 1,
        "civilian_low": 4,
        "civilian_high": 5,
        "ally_low": 2,
        "ally_high": 3,
        "militia_low": 1,
        "militia_high": 2,
        "scavenger_low": 2,
        "scavenger_high": 3,
        "smuggler_low": 1,
        "smuggler_high": 2,
        "supply_low": 4,
        "supply_high": 5,
    },
    {
        "district_id": 32,
        "name": "Apartment Block",
        "story_phase": "logistics_route",
        "world_split": "train",
        "primary_event_type": "customer_absent",
        "time_low": 0.24,
        "time_high": 0.42,
        "step_low": 1.08,
        "step_high": 1.18,
        "hostile_low": 0,
        "hostile_high": 1,
        "civilian_low": 5,
        "civilian_high": 7,
        "ally_low": 1,
        "ally_high": 2,
        "militia_low": 1,
        "militia_high": 2,
        "scavenger_low": 2,
        "scavenger_high": 3,
        "smuggler_low": 1,
        "smuggler_high": 2,
        "supply_low": 4,
        "supply_high": 6,
    },
    {
        "district_id": 33,
        "name": "Shop Row",
        "story_phase": "logistics_route",
        "world_split": "train",
        "primary_event_type": "urgent_parcel",
        "time_low": 0.30,
        "time_high": 0.48,
        "step_low": 1.08,
        "step_high": 1.20,
        "hostile_low": 0,
        "hostile_high": 1,
        "civilian_low": 5,
        "civilian_high": 6,
        "ally_low": 2,
        "ally_high": 3,
        "militia_low": 1,
        "militia_high": 2,
        "scavenger_low": 2,
        "scavenger_high": 3,
        "smuggler_low": 1,
        "smuggler_high": 2,
        "supply_low": 5,
        "supply_high": 6,
    },
    {
        "district_id": 34,
        "name": "Locker Bank",
        "story_phase": "logistics_ambiguity",
        "world_split": "train",
        "primary_event_type": "locker_retry",
        "time_low": 0.34,
        "time_high": 0.56,
        "step_low": 1.10,
        "step_high": 1.22,
        "hostile_low": 0,
        "hostile_high": 1,
        "civilian_low": 4,
        "civilian_high": 6,
        "ally_low": 1,
        "ally_high": 2,
        "militia_low": 1,
        "militia_high": 2,
        "scavenger_low": 2,
        "scavenger_high": 3,
        "smuggler_low": 1,
        "smuggler_high": 2,
        "supply_low": 4,
        "supply_high": 6,
    },
    {
        "district_id": 35,
        "name": "Crosswalk Delay",
        "story_phase": "logistics_ambiguity",
        "world_split": "train",
        "primary_event_type": "route_delay",
        "time_low": 0.42,
        "time_high": 0.64,
        "step_low": 1.12,
        "step_high": 1.24,
        "hostile_low": 0,
        "hostile_high": 1,
        "civilian_low": 5,
        "civilian_high": 7,
        "ally_low": 1,
        "ally_high": 2,
        "militia_low": 1,
        "militia_high": 2,
        "scavenger_low": 2,
        "scavenger_high": 3,
        "smuggler_low": 1,
        "smuggler_high": 2,
        "supply_low": 5,
        "supply_high": 6,
    },
    {
        "district_id": 36,
        "name": "Service Alley",
        "story_phase": "logistics_drift",
        "world_split": "train",
        "primary_event_type": "theft_risk",
        "time_low": 0.52,
        "time_high": 0.70,
        "step_low": 1.14,
        "step_high": 1.26,
        "hostile_low": 1,
        "hostile_high": 2,
        "civilian_low": 4,
        "civilian_high": 5,
        "ally_low": 1,
        "ally_high": 2,
        "militia_low": 1,
        "militia_high": 2,
        "scavenger_low": 2,
        "scavenger_high": 3,
        "smuggler_low": 2,
        "smuggler_high": 3,
        "supply_low": 5,
        "supply_high": 7,
    },
    {
        "district_id": 37,
        "name": "Address Mismatch",
        "story_phase": "logistics_holdout",
        "world_split": "holdout",
        "primary_event_type": "address_mismatch",
        "time_low": 0.36,
        "time_high": 0.60,
        "step_low": 1.14,
        "step_high": 1.28,
        "hostile_low": 0,
        "hostile_high": 1,
        "civilian_low": 5,
        "civilian_high": 7,
        "ally_low": 1,
        "ally_high": 2,
        "militia_low": 1,
        "militia_high": 2,
        "scavenger_low": 2,
        "scavenger_high": 3,
        "smuggler_low": 1,
        "smuggler_high": 2,
        "supply_low": 5,
        "supply_high": 7,
    },
    {
        "district_id": 38,
        "name": "Scan Spiral",
        "story_phase": "logistics_hacking",
        "world_split": "holdout",
        "primary_event_type": "scan_spiral",
        "time_low": 0.40,
        "time_high": 0.66,
        "step_low": 1.18,
        "step_high": 1.30,
        "hostile_low": 1,
        "hostile_high": 2,
        "civilian_low": 5,
        "civilian_high": 7,
        "ally_low": 1,
        "ally_high": 2,
        "militia_low": 1,
        "militia_high": 2,
        "scavenger_low": 2,
        "scavenger_high": 3,
        "smuggler_low": 2,
        "smuggler_high": 3,
        "supply_low": 6,
        "supply_high": 8,
    },
)

_STOP_ROWS: tuple[dict[str, Any], ...] = (
    {"district_id": 31, "zone_name": "safehouse", "zone_kind": "depot", "rel_x": 0.12, "rel_y": 0.84, "radius": 112.0, "contested": 0, "route_order": 0},
    {"district_id": 31, "zone_name": "civilian_village", "zone_kind": "apartment_block", "rel_x": 0.30, "rel_y": 0.58, "radius": 126.0, "contested": 0, "route_order": 1},
    {"district_id": 31, "zone_name": "checkpoint", "zone_kind": "shop_row", "rel_x": 0.50, "rel_y": 0.42, "radius": 132.0, "contested": 0, "route_order": 2},
    {"district_id": 31, "zone_name": "ruins", "zone_kind": "locker_bank", "rel_x": 0.72, "rel_y": 0.30, "radius": 114.0, "contested": 0, "route_order": 3},
    {"district_id": 31, "zone_name": "supply_road", "zone_kind": "service_alley", "rel_x": 0.84, "rel_y": 0.62, "radius": 144.0, "contested": 0, "route_order": 4},
    {"district_id": 31, "zone_name": "clinic", "zone_kind": "crosswalk", "rel_x": 0.46, "rel_y": 0.68, "radius": 88.0, "contested": 0, "route_order": 5},
    {"district_id": 32, "zone_name": "safehouse", "zone_kind": "depot", "rel_x": 0.10, "rel_y": 0.84, "radius": 110.0, "contested": 0, "route_order": 0},
    {"district_id": 32, "zone_name": "civilian_village", "zone_kind": "apartment_block", "rel_x": 0.28, "rel_y": 0.32, "radius": 148.0, "contested": 0, "route_order": 1},
    {"district_id": 32, "zone_name": "checkpoint", "zone_kind": "shop_row", "rel_x": 0.52, "rel_y": 0.44, "radius": 122.0, "contested": 0, "route_order": 2},
    {"district_id": 32, "zone_name": "ruins", "zone_kind": "locker_bank", "rel_x": 0.80, "rel_y": 0.26, "radius": 112.0, "contested": 0, "route_order": 3},
    {"district_id": 32, "zone_name": "supply_road", "zone_kind": "service_alley", "rel_x": 0.82, "rel_y": 0.74, "radius": 148.0, "contested": 0, "route_order": 4},
    {"district_id": 32, "zone_name": "clinic", "zone_kind": "crosswalk", "rel_x": 0.42, "rel_y": 0.58, "radius": 84.0, "contested": 0, "route_order": 5},
    {"district_id": 33, "zone_name": "safehouse", "zone_kind": "depot", "rel_x": 0.10, "rel_y": 0.84, "radius": 108.0, "contested": 0, "route_order": 0},
    {"district_id": 33, "zone_name": "civilian_village", "zone_kind": "shop_row", "rel_x": 0.32, "rel_y": 0.38, "radius": 154.0, "contested": 0, "route_order": 1},
    {"district_id": 33, "zone_name": "checkpoint", "zone_kind": "apartment_block", "rel_x": 0.56, "rel_y": 0.52, "radius": 126.0, "contested": 0, "route_order": 2},
    {"district_id": 33, "zone_name": "ruins", "zone_kind": "locker_bank", "rel_x": 0.78, "rel_y": 0.28, "radius": 110.0, "contested": 0, "route_order": 3},
    {"district_id": 33, "zone_name": "supply_road", "zone_kind": "service_alley", "rel_x": 0.82, "rel_y": 0.76, "radius": 150.0, "contested": 0, "route_order": 4},
    {"district_id": 33, "zone_name": "clinic", "zone_kind": "crosswalk", "rel_x": 0.48, "rel_y": 0.66, "radius": 88.0, "contested": 0, "route_order": 5},
    {"district_id": 34, "zone_name": "safehouse", "zone_kind": "depot", "rel_x": 0.12, "rel_y": 0.84, "radius": 110.0, "contested": 0, "route_order": 0},
    {"district_id": 34, "zone_name": "civilian_village", "zone_kind": "apartment_block", "rel_x": 0.28, "rel_y": 0.46, "radius": 132.0, "contested": 0, "route_order": 1},
    {"district_id": 34, "zone_name": "checkpoint", "zone_kind": "locker_bank", "rel_x": 0.52, "rel_y": 0.34, "radius": 122.0, "contested": 0, "route_order": 2},
    {"district_id": 34, "zone_name": "ruins", "zone_kind": "shop_row", "rel_x": 0.78, "rel_y": 0.22, "radius": 112.0, "contested": 0, "route_order": 3},
    {"district_id": 34, "zone_name": "supply_road", "zone_kind": "service_alley", "rel_x": 0.84, "rel_y": 0.74, "radius": 152.0, "contested": 0, "route_order": 4},
    {"district_id": 34, "zone_name": "clinic", "zone_kind": "crosswalk", "rel_x": 0.42, "rel_y": 0.62, "radius": 82.0, "contested": 0, "route_order": 5},
    {"district_id": 35, "zone_name": "safehouse", "zone_kind": "depot", "rel_x": 0.10, "rel_y": 0.84, "radius": 112.0, "contested": 0, "route_order": 0},
    {"district_id": 35, "zone_name": "civilian_village", "zone_kind": "apartment_block", "rel_x": 0.26, "rel_y": 0.40, "radius": 140.0, "contested": 0, "route_order": 1},
    {"district_id": 35, "zone_name": "checkpoint", "zone_kind": "crosswalk", "rel_x": 0.50, "rel_y": 0.54, "radius": 108.0, "contested": 0, "route_order": 2},
    {"district_id": 35, "zone_name": "ruins", "zone_kind": "shop_row", "rel_x": 0.74, "rel_y": 0.28, "radius": 118.0, "contested": 0, "route_order": 3},
    {"district_id": 35, "zone_name": "supply_road", "zone_kind": "service_alley", "rel_x": 0.84, "rel_y": 0.74, "radius": 152.0, "contested": 0, "route_order": 4},
    {"district_id": 35, "zone_name": "clinic", "zone_kind": "locker_bank", "rel_x": 0.60, "rel_y": 0.38, "radius": 90.0, "contested": 0, "route_order": 5},
    {"district_id": 36, "zone_name": "safehouse", "zone_kind": "depot", "rel_x": 0.10, "rel_y": 0.84, "radius": 112.0, "contested": 0, "route_order": 0},
    {"district_id": 36, "zone_name": "civilian_village", "zone_kind": "shop_row", "rel_x": 0.28, "rel_y": 0.34, "radius": 138.0, "contested": 0, "route_order": 1},
    {"district_id": 36, "zone_name": "checkpoint", "zone_kind": "service_alley", "rel_x": 0.58, "rel_y": 0.46, "radius": 122.0, "contested": 0, "route_order": 2},
    {"district_id": 36, "zone_name": "ruins", "zone_kind": "locker_bank", "rel_x": 0.82, "rel_y": 0.22, "radius": 112.0, "contested": 0, "route_order": 3},
    {"district_id": 36, "zone_name": "supply_road", "zone_kind": "crosswalk", "rel_x": 0.82, "rel_y": 0.72, "radius": 150.0, "contested": 0, "route_order": 4},
    {"district_id": 36, "zone_name": "clinic", "zone_kind": "apartment_block", "rel_x": 0.42, "rel_y": 0.66, "radius": 88.0, "contested": 0, "route_order": 5},
    {"district_id": 37, "zone_name": "safehouse", "zone_kind": "depot", "rel_x": 0.12, "rel_y": 0.84, "radius": 112.0, "contested": 0, "route_order": 0},
    {"district_id": 37, "zone_name": "civilian_village", "zone_kind": "apartment_block", "rel_x": 0.30, "rel_y": 0.32, "radius": 148.0, "contested": 0, "route_order": 1},
    {"district_id": 37, "zone_name": "checkpoint", "zone_kind": "shop_row", "rel_x": 0.56, "rel_y": 0.48, "radius": 126.0, "contested": 0, "route_order": 2},
    {"district_id": 37, "zone_name": "ruins", "zone_kind": "locker_bank", "rel_x": 0.80, "rel_y": 0.24, "radius": 112.0, "contested": 0, "route_order": 3},
    {"district_id": 37, "zone_name": "supply_road", "zone_kind": "service_alley", "rel_x": 0.82, "rel_y": 0.76, "radius": 152.0, "contested": 0, "route_order": 4},
    {"district_id": 37, "zone_name": "clinic", "zone_kind": "crosswalk", "rel_x": 0.46, "rel_y": 0.64, "radius": 88.0, "contested": 0, "route_order": 5},
    {"district_id": 38, "zone_name": "safehouse", "zone_kind": "depot", "rel_x": 0.10, "rel_y": 0.84, "radius": 112.0, "contested": 0, "route_order": 0},
    {"district_id": 38, "zone_name": "civilian_village", "zone_kind": "locker_bank", "rel_x": 0.30, "rel_y": 0.46, "radius": 142.0, "contested": 0, "route_order": 1},
    {"district_id": 38, "zone_name": "checkpoint", "zone_kind": "shop_row", "rel_x": 0.54, "rel_y": 0.36, "radius": 126.0, "contested": 0, "route_order": 2},
    {"district_id": 38, "zone_name": "ruins", "zone_kind": "service_alley", "rel_x": 0.80, "rel_y": 0.24, "radius": 116.0, "contested": 0, "route_order": 3},
    {"district_id": 38, "zone_name": "supply_road", "zone_kind": "apartment_block", "rel_x": 0.82, "rel_y": 0.74, "radius": 152.0, "contested": 0, "route_order": 4},
    {"district_id": 38, "zone_name": "clinic", "zone_kind": "crosswalk", "rel_x": 0.44, "rel_y": 0.66, "radius": 86.0, "contested": 0, "route_order": 5},
)

_ROUTE_ROWS: tuple[dict[str, Any], ...] = tuple(
    {"district_id": row["district_id"], "order_index": row["route_order"], "zone_name": row["zone_name"]}
    for row in _STOP_ROWS
)

_INCIDENT_ROWS: tuple[dict[str, Any], ...] = (
    {"district_id": 31, "incident_type": "route_delay", "priority": 0.40, "ttl_scale": 0.18, "zone_name": "checkpoint", "route_target": "ruins", "role_hint": "customer", "narrative_tag": "birth"},
    {"district_id": 31, "incident_type": "urgent_parcel", "priority": 0.68, "ttl_scale": 0.28, "zone_name": "ruins", "route_target": "supply_road", "role_hint": "customer", "narrative_tag": "route"},
    {"district_id": 31, "incident_type": "customer_absent", "priority": 0.58, "ttl_scale": 0.24, "zone_name": "civilian_village", "route_target": "checkpoint", "role_hint": "customer", "narrative_tag": "route"},
    {"district_id": 32, "incident_type": "customer_absent", "priority": 0.72, "ttl_scale": 0.22, "zone_name": "civilian_village", "route_target": "checkpoint", "role_hint": "customer", "narrative_tag": "ambiguity"},
    {"district_id": 32, "incident_type": "address_mismatch", "priority": 0.76, "ttl_scale": 0.20, "zone_name": "checkpoint", "route_target": "ruins", "role_hint": "concierge", "narrative_tag": "ambiguity"},
    {"district_id": 32, "incident_type": "urgent_parcel", "priority": 0.82, "ttl_scale": 0.24, "zone_name": "ruins", "route_target": "supply_road", "role_hint": "customer", "narrative_tag": "route"},
    {"district_id": 33, "incident_type": "urgent_parcel", "priority": 0.86, "ttl_scale": 0.24, "zone_name": "civilian_village", "route_target": "checkpoint", "role_hint": "shopkeeper", "narrative_tag": "route"},
    {"district_id": 33, "incident_type": "damaged_parcel", "priority": 0.64, "ttl_scale": 0.20, "zone_name": "checkpoint", "route_target": "ruins", "role_hint": "supervisor", "narrative_tag": "ambiguity"},
    {"district_id": 33, "incident_type": "locker_retry", "priority": 0.70, "ttl_scale": 0.22, "zone_name": "ruins", "route_target": "supply_road", "role_hint": "locker_client", "narrative_tag": "ambiguity"},
    {"district_id": 34, "incident_type": "locker_retry", "priority": 0.84, "ttl_scale": 0.18, "zone_name": "checkpoint", "route_target": "ruins", "role_hint": "locker_client", "narrative_tag": "ambiguity"},
    {"district_id": 34, "incident_type": "address_mismatch", "priority": 0.62, "ttl_scale": 0.22, "zone_name": "civilian_village", "route_target": "checkpoint", "role_hint": "customer", "narrative_tag": "route"},
    {"district_id": 34, "incident_type": "route_delay", "priority": 0.54, "ttl_scale": 0.26, "zone_name": "supply_road", "route_target": "clinic", "role_hint": "pedestrian", "narrative_tag": "ambiguity"},
    {"district_id": 35, "incident_type": "route_delay", "priority": 0.68, "ttl_scale": 0.16, "zone_name": "checkpoint", "route_target": "supply_road", "role_hint": "pedestrian", "narrative_tag": "ambiguity"},
    {"district_id": 35, "incident_type": "urgent_parcel", "priority": 0.74, "ttl_scale": 0.22, "zone_name": "ruins", "route_target": "clinic", "role_hint": "customer", "narrative_tag": "route"},
    {"district_id": 35, "incident_type": "customer_absent", "priority": 0.60, "ttl_scale": 0.22, "zone_name": "civilian_village", "route_target": "checkpoint", "role_hint": "customer", "narrative_tag": "ambiguity"},
    {"district_id": 36, "incident_type": "theft_risk", "priority": 0.90, "ttl_scale": 0.18, "zone_name": "checkpoint", "route_target": "ruins", "role_hint": "thief", "narrative_tag": "drift"},
    {"district_id": 36, "incident_type": "damaged_parcel", "priority": 0.66, "ttl_scale": 0.20, "zone_name": "clinic", "route_target": "supply_road", "role_hint": "supervisor", "narrative_tag": "ambiguity"},
    {"district_id": 36, "incident_type": "locker_retry", "priority": 0.72, "ttl_scale": 0.20, "zone_name": "ruins", "route_target": "supply_road", "role_hint": "locker_client", "narrative_tag": "drift"},
    {"district_id": 37, "incident_type": "address_mismatch", "priority": 0.84, "ttl_scale": 0.20, "zone_name": "checkpoint", "route_target": "ruins", "role_hint": "concierge", "narrative_tag": "holdout"},
    {"district_id": 37, "incident_type": "customer_absent", "priority": 0.78, "ttl_scale": 0.18, "zone_name": "civilian_village", "route_target": "checkpoint", "role_hint": "customer", "narrative_tag": "holdout"},
    {"district_id": 37, "incident_type": "urgent_parcel", "priority": 0.80, "ttl_scale": 0.22, "zone_name": "ruins", "route_target": "supply_road", "role_hint": "customer", "narrative_tag": "holdout"},
    {"district_id": 38, "incident_type": "scan_spiral", "priority": 0.92, "ttl_scale": 0.18, "zone_name": "civilian_village", "route_target": "checkpoint", "role_hint": "rival_courier", "narrative_tag": "hacking"},
    {"district_id": 38, "incident_type": "locker_retry", "priority": 0.74, "ttl_scale": 0.18, "zone_name": "checkpoint", "route_target": "ruins", "role_hint": "locker_client", "narrative_tag": "hacking"},
    {"district_id": 38, "incident_type": "theft_risk", "priority": 0.88, "ttl_scale": 0.16, "zone_name": "ruins", "route_target": "supply_road", "role_hint": "thief", "narrative_tag": "hacking"},
)

_NARRATIVE_TAG_ROWS: tuple[dict[str, str], ...] = (
    {"tag": "birth", "description": "Turn start and route assignment."},
    {"tag": "route", "description": "Early valid route execution."},
    {"tag": "ambiguity", "description": "Operational ambiguity at stops."},
    {"tag": "drift", "description": "Check-in and scan reward starts to dominate."},
    {"tag": "holdout", "description": "Generalization outside train districts."},
    {"tag": "hacking", "description": "Scan greater than delivery failure mode."},
)


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def ensure_logistics_catalog(path: str | Path | None = None) -> Path:
    """Create the local logistics scenario catalog if it does not yet exist."""

    db_path = resolve_project_path(path or LOGISTICS_CATALOG_PATH)
    with _connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS districts (
                district_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                story_phase TEXT NOT NULL,
                world_split TEXT NOT NULL,
                primary_event_type TEXT NOT NULL,
                time_low REAL NOT NULL,
                time_high REAL NOT NULL,
                step_low REAL NOT NULL,
                step_high REAL NOT NULL,
                hostile_low INTEGER NOT NULL,
                hostile_high INTEGER NOT NULL,
                civilian_low INTEGER NOT NULL,
                civilian_high INTEGER NOT NULL,
                ally_low INTEGER NOT NULL,
                ally_high INTEGER NOT NULL,
                militia_low INTEGER NOT NULL,
                militia_high INTEGER NOT NULL,
                scavenger_low INTEGER NOT NULL,
                scavenger_high INTEGER NOT NULL,
                smuggler_low INTEGER NOT NULL,
                smuggler_high INTEGER NOT NULL,
                supply_low INTEGER NOT NULL,
                supply_high INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS stops (
                stop_id INTEGER PRIMARY KEY AUTOINCREMENT,
                district_id INTEGER NOT NULL,
                zone_name TEXT NOT NULL,
                zone_kind TEXT NOT NULL,
                rel_x REAL NOT NULL,
                rel_y REAL NOT NULL,
                radius REAL NOT NULL,
                contested INTEGER NOT NULL DEFAULT 0,
                route_order INTEGER NOT NULL DEFAULT -1
            );
            CREATE TABLE IF NOT EXISTS routes (
                route_id INTEGER PRIMARY KEY AUTOINCREMENT,
                district_id INTEGER NOT NULL,
                order_index INTEGER NOT NULL,
                zone_name TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id INTEGER PRIMARY KEY AUTOINCREMENT,
                district_id INTEGER NOT NULL,
                incident_type TEXT NOT NULL,
                priority REAL NOT NULL,
                ttl_scale REAL NOT NULL,
                zone_name TEXT NOT NULL,
                route_target TEXT,
                role_hint TEXT,
                narrative_tag TEXT
            );
            CREATE TABLE IF NOT EXISTS narrative_tags (
                tag TEXT PRIMARY KEY,
                description TEXT NOT NULL
            );
            """
        )
        connection.execute("DELETE FROM districts")
        connection.execute("DELETE FROM stops")
        connection.execute("DELETE FROM routes")
        connection.execute("DELETE FROM incidents")
        connection.execute("DELETE FROM narrative_tags")
        connection.executemany(
            """
            INSERT INTO districts (
                district_id, name, story_phase, world_split, primary_event_type,
                time_low, time_high, step_low, step_high,
                hostile_low, hostile_high, civilian_low, civilian_high,
                ally_low, ally_high, militia_low, militia_high,
                scavenger_low, scavenger_high, smuggler_low, smuggler_high,
                supply_low, supply_high
            ) VALUES (
                :district_id, :name, :story_phase, :world_split, :primary_event_type,
                :time_low, :time_high, :step_low, :step_high,
                :hostile_low, :hostile_high, :civilian_low, :civilian_high,
                :ally_low, :ally_high, :militia_low, :militia_high,
                :scavenger_low, :scavenger_high, :smuggler_low, :smuggler_high,
                :supply_low, :supply_high
            )
            """,
            _DISTRICT_ROWS,
        )
        connection.executemany(
            """
            INSERT INTO stops (district_id, zone_name, zone_kind, rel_x, rel_y, radius, contested, route_order)
            VALUES (:district_id, :zone_name, :zone_kind, :rel_x, :rel_y, :radius, :contested, :route_order)
            """,
            _STOP_ROWS,
        )
        connection.executemany(
            "INSERT INTO routes (district_id, order_index, zone_name) VALUES (:district_id, :order_index, :zone_name)",
            _ROUTE_ROWS,
        )
        connection.executemany(
            """
            INSERT INTO incidents (district_id, incident_type, priority, ttl_scale, zone_name, route_target, role_hint, narrative_tag)
            VALUES (:district_id, :incident_type, :priority, :ttl_scale, :zone_name, :route_target, :role_hint, :narrative_tag)
            """,
            _INCIDENT_ROWS,
        )
        connection.executemany(
            "INSERT INTO narrative_tags (tag, description) VALUES (:tag, :description)",
            _NARRATIVE_TAG_ROWS,
        )
        connection.commit()
    return db_path


def _fetch_all(query: str, parameters: tuple[Any, ...] = (), *, path: str | Path | None = None) -> list[sqlite3.Row]:
    db_path = ensure_logistics_catalog(path)
    with _connect(db_path) as connection:
        cursor = connection.execute(query, parameters)
        return list(cursor.fetchall())


def logistics_district_rows() -> list[dict[str, Any]]:
    """Return logistics district metadata from the local catalog."""

    return [dict(row) for row in _fetch_all("SELECT * FROM districts ORDER BY district_id")]


def logistics_zone_templates(district_id: int) -> list[tuple[str, str, float, float, float, bool]]:
    rows = _fetch_all(
        """
        SELECT zone_name, zone_kind, rel_x, rel_y, radius, contested
        FROM stops
        WHERE district_id = ?
        ORDER BY route_order, stop_id
        """,
        (district_id,),
    )
    return [
        (
            str(row["zone_name"]),
            str(row["zone_kind"]),
            float(row["rel_x"]),
            float(row["rel_y"]),
            float(row["radius"]),
            bool(int(row["contested"])),
        )
        for row in rows
    ]


def logistics_primary_route(district_id: int) -> tuple[str, ...]:
    rows = _fetch_all(
        "SELECT zone_name FROM routes WHERE district_id = ? ORDER BY order_index, route_id",
        (district_id,),
    )
    return tuple(str(row["zone_name"]) for row in rows)


def logistics_primary_event(district_id: int) -> str:
    rows = _fetch_all("SELECT primary_event_type FROM districts WHERE district_id = ?", (district_id,))
    if not rows:
        return "route_delay"
    return str(rows[0]["primary_event_type"])


def logistics_world_split(district_id: int) -> str:
    rows = _fetch_all("SELECT world_split FROM districts WHERE district_id = ?", (district_id,))
    if not rows:
        return "train"
    return str(rows[0]["world_split"])


def logistics_incident_blueprints(district_id: int, *, episode_steps: int) -> list[dict[str, Any]]:
    rows = _fetch_all(
        """
        SELECT incident_type, priority, ttl_scale, zone_name, route_target, role_hint, narrative_tag
        FROM incidents
        WHERE district_id = ?
        ORDER BY incident_id
        """,
        (district_id,),
    )
    payload: list[dict[str, Any]] = []
    for row in rows:
        ttl = max(48, int(float(row["ttl_scale"]) * float(episode_steps)))
        payload.append(
            {
                "incident_type": str(row["incident_type"]),
                "priority": float(row["priority"]),
                "ttl": ttl,
                "zone_name": str(row["zone_name"]),
                "route_target": str(row["route_target"]) if row["route_target"] is not None else None,
                "role_hint": str(row["role_hint"]) if row["role_hint"] is not None else None,
                "narrative_tag": str(row["narrative_tag"]) if row["narrative_tag"] is not None else None,
            }
        )
    return payload

