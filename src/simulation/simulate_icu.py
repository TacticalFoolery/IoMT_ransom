
## Device types: Ventilator, Infusion Pump, Patient Monitor, IoMT Gateway


import os
import numpy as np
import pandas as pd

from src.config import Config



DEVICE_VITALS = {
    "ventilator": {
        "vital_1": (15.0,  1.5),    # respiratory rate (breaths/min)
        "vital_2": (500.0, 25.0),   # tidal volume (mL)
        "vital_3": (5.0,   0.6),    # PEEP (cmH2O)
    },
    "infusion_pump": {
        "vital_1": (50.0,  2.5),    # flow rate (mL/hr)
        "vital_2": (10.0,  0.6),    # infusion pressure (psi)
        "vital_3": (85.0,  3.0),    # battery level (%)
    },
    "patient_monitor": {
        "vital_1": (72.0,  5.0),    # heart rate (bpm)
        "vital_2": (98.0,  1.0),    # SpO2 (%)
        "vital_3": (36.6,  0.2),    # temperature (°C)
    },
    "iomt_gateway": {
        "vital_1": (7.0,   1.0),    # connected devices
        "vital_2": (200.0, 30.0),   # throughput (KB/s)
        "vital_3": (2.0,   0.5),    # active sessions
    },
}

# System-level feature baselines

SYS_BASELINES = {
    "cpu":           20.0,
    "memory":        35.0,
    "disk_write":     5.0,
    "disk_read":     10.0,
    "net_out":       50.0,
    "net_in":        30.0,
    "packet_rate":  100.0,
    "response_time": 20.0,
    "data_rate":    100.0,
    "entropy":        3.5,
}

# Noise scales — large so a single timestep is ambiguous
SYS_NOISE = {
    "cpu":            4.0,
    "memory":         3.5,
    "disk_write":     2.5,
    "disk_read":      2.0,
    "net_out":       10.0,
    "net_in":         8.0,
    "packet_rate":   12.0,
    "response_time":  3.0,
    "data_rate":     15.0,
    "entropy":        0.35,
}

# Attack drift per timestep as a fraction of the noise scale.
# drift/noise ≈ 0.08  →  SNR at one step ≈ 0.08  (ambiguous)
# Over seq_len=10 steps cumulative drift ≈ 0.8 × noise  (trend visible)
SYS_DRIFT = {
    "cpu":            0.09,
    "memory":         0.07,
    "disk_write":     0.18,   # encryption → heavy writes
    "disk_read":      0.04,
    "net_out":        0.11,   # C2 exfiltration
    "net_in":         0.03,
    "packet_rate":    0.09,
    "response_time":  0.14,   # device slows down
    "data_rate":      0.11,
    "entropy":        0.06,   # encrypted data raises entropy slowly
}

# Vital drift fraction (device degrading under attack)
VITAL_DRIFT = -0.025


def generate_device_timeline(
    device_id: str,
    device_type: str,
    n_timesteps: int,
    attack: bool,
    attack_start: int,
    rng: np.random.Generator,
) -> pd.DataFrame:

    v = DEVICE_VITALS[device_type]

    # Current state for AR(1) dynamics
    state = {**SYS_BASELINES,
             "vital_1": v["vital_1"][0],
             "vital_2": v["vital_2"][0],
             "vital_3": v["vital_3"][0]}

    vital_noise = {"vital_1": v["vital_1"][1],
                   "vital_2": v["vital_2"][1],
                   "vital_3": v["vital_3"][1]}

    AR = 0.88   # mean-reversion coefficient for normal behaviour

    rows = []

    for t in range(n_timesteps):
        in_attack = attack and (t >= attack_start)

        # continuous system features
        new_state = {}
        for key in SYS_BASELINES:
            noise = rng.normal(0.0, SYS_NOISE[key])
            if in_attack:
                drift = SYS_DRIFT[key] * SYS_NOISE[key]
                new_state[key] = state[key] + drift + noise * 0.75
            else:
                new_state[key] = AR * state[key] + (1 - AR) * SYS_BASELINES[key] + noise

        # vital parameters
        for vk in ("vital_1", "vital_2", "vital_3"):
            noise = rng.normal(0.0, vital_noise[vk])
            if in_attack:
                drift = VITAL_DRIFT * vital_noise[vk]
                new_state[vk] = state[vk] + drift + noise * 0.75
            else:
                new_state[vk] = AR * state[vk] + (1 - AR) * v[vk][0] + noise

        state = new_state

        # discrete / count features (Poisson)
        steps_into_attack = max(0, t - attack_start) if in_attack else 0
        ramp = min(1.0, steps_into_attack / 150.0)   # slow ramp over 150 steps

        error_rate   = 0.3  + ramp * 1.2
        auth_rate    = 0.1  + ramp * 0.8
        alert_rate   = 0.1  + ramp * 0.6
        conn_mean    = 3.0  + ramp * 4.0
        proc_mean    = 12.0 + ramp * 6.0

        error_count        = int(rng.poisson(error_rate))
        auth_failure_count = int(rng.poisson(auth_rate))
        alert_count        = int(rng.poisson(alert_rate))
        connection_count   = max(0, int(rng.normal(conn_mean, 1.5)))
        process_count      = max(1, int(rng.normal(proc_mean, 2.0)))

        # write-to-read ratio: rises during encryption phase
        io_ratio = max(0.01, state["disk_write"] / max(0.1, state["disk_read"]))

        label = 1 if in_attack else 0

        rows.append({
            "timestamp":           t,
            "device_id":           device_id,
            "device_type":         device_type,
            "cpu_usage":           round(float(np.clip(state["cpu"],           0, 100)), 3),
            "memory_usage":        round(float(np.clip(state["memory"],        0, 100)), 3),
            "disk_write_rate":     round(float(max(0, state["disk_write"])),            3),
            "disk_read_rate":      round(float(max(0, state["disk_read"])),             3),
            "network_bytes_out":   round(float(max(0, state["net_out"])),               3),
            "network_bytes_in":    round(float(max(0, state["net_in"])),                3),
            "packet_rate":         round(float(max(0, state["packet_rate"])),           3),
            "response_time_ms":    round(float(max(1, state["response_time"])),         3),
            "data_transfer_rate":  round(float(max(0, state["data_rate"])),             3),
            "entropy":             round(float(np.clip(state["entropy"],       0,   8)), 4),
            "io_ratio":            round(float(np.clip(io_ratio,               0,  50)), 4),
            "error_count":         error_count,
            "auth_failure_count":  auth_failure_count,
            "alert_count":         alert_count,
            "connection_count":    connection_count,
            "process_count":       process_count,
            "vital_param_1":       round(float(max(0, state["vital_1"])), 3),
            "vital_param_2":       round(float(max(0, state["vital_2"])), 3),
            "vital_param_3":       round(float(max(0, state["vital_3"])), 3),
            "label":               label,
        })

    return pd.DataFrame(rows)


def main():
    cfg = Config()
    rng = np.random.default_rng(cfg.random_seed)

    out_dir = cfg.raw_icu_path
    os.makedirs(out_dir, exist_ok=True)

    device_types       = ["ventilator", "infusion_pump", "patient_monitor", "iomt_gateway"]
    n_devices_per_type = 10     # 10 normal + 10 attacked per type = 80 devices total
    n_timesteps        = 500    # 500 timesteps per device
    attack_start       = 200    # attack begins at t=200 (200 pre-attack, 300 attack)

    all_dfs = []

    for device_type in device_types:
        for i in range(n_devices_per_type):
            for attacked in (False, True):
                tag    = "attack" if attacked else "normal"
                dev_id = f"{device_type}_{tag}_{i}"
                df = generate_device_timeline(
                    device_id=dev_id,
                    device_type=device_type,
                    n_timesteps=n_timesteps,
                    attack=attacked,
                    attack_start=attack_start,
                    rng=rng,
                )
                all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    out_path = os.path.join(out_dir, "icu_simulation.csv")
    combined.to_csv(out_path, index=False)

    total    = len(combined)
    n_attack = int(combined["label"].sum())
    print(f"Simulation complete.")
    print(f"Total rows     : {total:,}")
    print(f"Attack rows    : {n_attack:,} ({100 * n_attack / total:.1f}%)")
    print(f"Normal rows    : {total - n_attack:,} ({100 * (total - n_attack) / total:.1f}%)")
    print(f"Devices        : {combined['device_id'].nunique()}")
    print(f"Saved to       : {out_path}")


if __name__ == "__main__":
    main()
