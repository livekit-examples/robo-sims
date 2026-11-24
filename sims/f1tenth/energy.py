"""Estimated electrical energy for an RC pack; simulation-time accounting."""
class Battery:
    def __init__(self, tuning):
        self.nominal_voltage = tuning["battery_nominal_voltage"]
        self.capacity_wh = self.nominal_voltage * tuning["battery_capacity_ah"]
        self.remaining_wh = self.capacity_wh
        self.idle_power_w = tuning["battery_idle_power_w"]
        self.efficiency = tuning["drive_efficiency"]
        self.full_torque_loss_w = tuning["drive_loss_at_full_torque_w"]

    def step(self, dt, positive_shaft_power_w, drive_load):
        # Braking dissipates energy: this RC model does not assume regeneration.
        watts = (self.idle_power_w + max(0.0, positive_shaft_power_w) / self.efficiency
                 + self.full_torque_loss_w * min(1.0, max(0.0, drive_load)) ** 2)
        self.remaining_wh = max(0.0, min(self.capacity_wh, self.remaining_wh - watts * dt / 3600))

    def state(self):
        return {"remaining_wh": self.remaining_wh, "capacity_wh": self.capacity_wh,
                "state_of_charge_percent": 100 * self.remaining_wh / self.capacity_wh}

    def voltage(self):
        # A simple state-of-charge estimate, not a fitted LiPo discharge curve.
        cells = self.nominal_voltage / 3.7
        return cells * (3.3 + 0.9 * self.remaining_wh / self.capacity_wh)
