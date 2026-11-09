"""Small, independently selectable instrument tracks for Teleop widgets."""
import math

SPEED_TRACK = "/speed_mph"
GEAR_TRACK = "/gear"
BATTERY_TRACK = "/battery_energy"
STEERING_TRACK = "/steering_angle_deg"
YAW_RATE_TRACK = "/yaw_rate_deg_s"
LONGITUDINAL_G_TRACK = "/longitudinal_acceleration_g"
LATERAL_G_TRACK = "/lateral_acceleration_g"
LAP_TRACK = "/lap_number"
LAP_TIME_TRACK = "/lap_time_s"
LAST_LAP_TRACK = "/last_valid_lap_time_s"
PROGRESS_TRACK = "/track_progress_percent"
FLOAT64 = "std_msgs/msg/Float64"
FLOAT64_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object", "required": ["data"], "additionalProperties": False,
    "properties": {"data": {"type": "number"}},
}
STRING = "std_msgs/msg/String"
STRING_SCHEMA = {**FLOAT64_SCHEMA, "properties": {"data": {"type": "string"}}}
INT32 = "std_msgs/msg/Int32"
INT32_SCHEMA = {**FLOAT64_SCHEMA, "properties": {"data": {
    "type": "integer", "minimum": -2147483648, "maximum": 2147483647}}}
OPTIONAL_FLOAT64 = "robo_sims/msg/OptionalFloat64V1"
OPTIONAL_FLOAT64_SCHEMA = {**FLOAT64_SCHEMA, "properties": {"data": {"type": ["number", "null"]}}}
BATTERY_ENERGY = "robo_sims/msg/BatteryEnergyV1"
BATTERY_ENERGY_SCHEMA = {**FLOAT64_SCHEMA,
    "required": ["remaining_wh", "capacity_wh", "state_of_charge_percent"],
    "properties": {"remaining_wh": {"type": "number", "minimum": 0},
                   "capacity_wh": {"type": "number", "exclusiveMinimum": 0},
                   "state_of_charge_percent": {"type": "number", "minimum": 0, "maximum": 100}}}
INSTRUMENT_SCHEMAS = {
    **{name: (FLOAT64, FLOAT64_SCHEMA) for name in (
        SPEED_TRACK, STEERING_TRACK, YAW_RATE_TRACK, LONGITUDINAL_G_TRACK,
        LATERAL_G_TRACK, LAP_TIME_TRACK, PROGRESS_TRACK)},
    GEAR_TRACK: (STRING, STRING_SCHEMA),
    BATTERY_TRACK: (BATTERY_ENERGY, BATTERY_ENERGY_SCHEMA),
    LAP_TRACK: (INT32, INT32_SCHEMA),
    LAST_LAP_TRACK: (OPTIONAL_FLOAT64, OPTIONAL_FLOAT64_SCHEMA),
}
MPH_PER_MPS = 3600 / 1609.344


def speed_mph(car):
    """Measured planar ground speed; reverse is positive, and sliding counts."""
    return {"data": math.hypot(car.vx, car.vy) * MPH_PER_MPS}


def instrument_value(car, name):
    if name == BATTERY_TRACK:
        return car.battery.state()
    values = {SPEED_TRACK: speed_mph(car)["data"], GEAR_TRACK: car.gear,
              STEERING_TRACK: math.degrees(car.steering),
              YAW_RATE_TRACK: math.degrees(car.yaw_rate),
              LONGITUDINAL_G_TRACK: car.ax / 9.81, LATERAL_G_TRACK: car.ay / 9.81,
              LAP_TRACK: car.laps, LAP_TIME_TRACK: car.time - car.lap_started,
              LAST_LAP_TRACK: car.last_lap_seconds,
              PROGRESS_TRACK: 100 * car.progress / car.track.length}
    return {"data": values[name]}
