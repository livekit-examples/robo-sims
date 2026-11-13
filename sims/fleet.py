"""Fleet Portal participant attributes (see docs/fleet-portal.md).

A simulation has no physical place, so each one reports a made-up location.
The robot, not the web page, owns these values and rewrites them when they change.
"""

from __future__ import annotations

MODEL, CONTROL = "lk.fleet.model", "lk.fleet.control"
CITY, COUNTRY, LAT, LNG = "lk.fleet.city", "lk.fleet.country", "lk.fleet.lat", "lk.fleet.lng"

# prefix -> (city, country, lat, lng). The first matching prefix wins.
LOCATIONS = (
    ("trossen-wxai-whack", ("Area 51", "US", 37.2431, -115.7930)),
    ("trossen-workbench-whack", ("Area 51", "US", 37.2431, -115.7930)),
    ("trossen-wxai", ("Tokyo", "JP", 35.6762, 139.6503)),
    ("trossen-workbench", ("Rio de Janeiro", "BR", -22.9068, -43.1729)),
    ("f1tenth", ("Gulf of Mexico", "US", 25.0, -90.0)),
    ("vacuum", ("San Francisco", "US", 37.7749, -122.4194)),
    ("cluttered-table", ("London", "GB", 51.5074, -0.1278)),
    ("carla", ("Manhattan", "US", 40.7831, -73.9712)),
)


def location_attributes(sim_name: str | None) -> dict[str, str]:
    for prefix, (city, country, lat, lng) in LOCATIONS:
        if sim_name and sim_name.startswith(prefix):
            return {CITY: city, COUNTRY: country, LAT: str(lat), LNG: str(lng)}
    return {}


def control_mode(active_operator: str, autonomous: bool = False) -> str:
    """One of the portal modes: autonomous, teleop or standby."""
    return "autonomous" if autonomous else "teleop" if active_operator else "standby"
