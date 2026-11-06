"""Virtual reBot Arm B601-DM hardware: MuJoCo physics behind an emulated Damiao bus (hardware level).

Clients talk to the simulated arm exactly as they talk to the real one, through
the Damiao USB2CAN serial bridge protocol, so the driver code is shared.
"""
