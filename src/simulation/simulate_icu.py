
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

# ── Ransomware variant definitions ──────────────────────────────────────────
# Label 0 = normal (no attack)
# Label 1 = encryption_heavy  — WannaCry / Ryuk style: encrypt-in-place
# Label 2 = exfiltration_first — Maze / REvil style: steal data, then encrypt
# Label 3 = wiper             — NotPetya style: overwrite / destroy data
# Label 4 = slow_burn         — LockBit stealth: low-and-slow, hard to detect

VARIANTS = {
    "encryption_heavy":   1,
    "exfiltration_first": 2,
    "wiper":              3,
    "slow_burn":          4,
}

# Per-variant drift per timestep as a fraction of SYS_NOISE.
# drift/noise ≈ 0.08 → SNR at one step ≈ 0.08 (ambiguous);
# over seq_len=20 steps the cumulative trend becomes visible.
VARIANT_DRIFT = {
    "encryption_heavy": {
        "cpu":           0.14,
        "memory":        0.10,
        "disk_write":    0.35,   # primary: heavy in-place encryption writes
        "disk_read":     0.05,
        "net_out":       0.08,   # C2 check-in / key exchange
        "net_in":        0.02,
        "packet_rate":   0.07,
        "response_time": 0.18,   # device slows under encryption load
        "data_rate":     0.08,
        "entropy":       0.12,   # encrypted files drive entropy up strongly
    },
    "exfiltration_first": {      # phase-1 (exfiltration) drift
        "cpu":           0.09,
        "memory":        0.08,
        "disk_write":    0.06,
        "disk_read":     0.16,   # reading files to stage for exfiltration
        "net_out":       0.28,   # primary: massive outbound data theft
        "net_in":        0.04,
        "packet_rate":   0.18,
        "response_time": 0.08,
        "data_rate":     0.22,
        "entropy":       0.03,   # raw data — entropy not elevated yet
    },
    "wiper": {
        "cpu":           0.16,
        "memory":        0.09,
        "disk_write":    0.30,   # overwriting disk sectors destructively
        "disk_read":     0.20,   # scanning filesystem before overwrite
        "net_out":       0.04,
        "net_in":        0.02,
        "packet_rate":   0.05,
        "response_time": 0.28,   # device degrades extremely fast
        "data_rate":     0.06,
        "entropy":       0.01,   # writing zeros / fixed patterns → low entropy
    },
    "slow_burn": {
        "cpu":           0.03,   # barely perceptible above noise floor
        "memory":        0.03,
        "disk_write":    0.06,
        "disk_read":     0.02,
        "net_out":       0.04,
        "net_in":        0.01,
        "packet_rate":   0.03,
        "response_time": 0.05,
        "data_rate":     0.04,
        "entropy":       0.04,   # very slow entropy climb
    },
}

# Phase-2 drift for exfiltration_first: switches to encryption after midpoint
EXFIL_PHASE2_DRIFT = {
    "cpu":           0.12,
    "memory":        0.09,
    "disk_write":    0.28,   # encryption begins
    "disk_read":     0.04,
    "net_out":       0.06,   # exfil traffic drops off
    "net_in":        0.02,
    "packet_rate":   0.06,
    "response_time": 0.15,
    "data_rate":     0.07,
    "entropy":       0.10,   # entropy climbs as files are encrypted
}

# How fast device vitals degrade per variant
VARIANT_VITAL_DRIFT = {
    "encryption_heavy":   -0.025,
    "exfiltration_first": -0.015,
    "wiper":              -0.060,   # catastrophic, device becomes non-functional fast
    "slow_burn":          -0.008,   # barely detectable degradation
}


def generate_device_timeline(
    device_id: str,
    device_type: str,
    n_timesteps: int,
    attack_variant: str | None,
    attack_start: int,
    rng: np.random.Generator,
) -> pd.DataFrame:

    v = DEVICE_VITALS[device_type]

    state = {**SYS_BASELINES,
             "vital_1": v["vital_1"][0],
             "vital_2": v["vital_2"][0],
             "vital_3": v["vital_3"][0]}

    vital_noise = {"vital_1": v["vital_1"][1],
                   "vital_2": v["vital_2"][1],
                   "vital_3": v["vital_3"][1]}

    AR = 0.88   # mean-reversion coefficient for normal behaviour

    attack_duration = n_timesteps - attack_start

    rows = []

    for t in range(n_timesteps):
        in_attack = attack_variant is not None and (t >= attack_start)

        if in_attack:
            if attack_variant == "exfiltration_first":
                # Switch from exfiltration to encryption at the midpoint
                phase2_start = attack_start + attack_duration // 2
                drift_profile = EXFIL_PHASE2_DRIFT if t >= phase2_start else VARIANT_DRIFT["exfiltration_first"]
            else:
                drift_profile = VARIANT_DRIFT[attack_variant]
            vital_drift = VARIANT_VITAL_DRIFT[attack_variant]
        else:
            drift_profile = None
            vital_drift   = 0.0

        # Continuous system features
        new_state = {}
        for key in SYS_BASELINES:
            noise = rng.normal(0.0, SYS_NOISE[key])
            if in_attack:
                drift = drift_profile[key] * SYS_NOISE[key]
                new_state[key] = state[key] + drift + noise * 0.75
            else:
                new_state[key] = AR * state[key] + (1 - AR) * SYS_BASELINES[key] + noise

        # Vital parameters
        for vk in ("vital_1", "vital_2", "vital_3"):
            noise = rng.normal(0.0, vital_noise[vk])
            if in_attack:
                drift = vital_drift * vital_noise[vk]
                new_state[vk] = state[vk] + drift + noise * 0.75
            else:
                new_state[vk] = AR * state[vk] + (1 - AR) * v[vk][0] + noise

        state = new_state

        # Discrete / count features (Poisson)
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

        io_ratio = max(0.01, state["disk_write"] / max(0.1, state["disk_read"]))

        label = VARIANTS[attack_variant] if in_attack else 0

        rows.append({
            "timestamp":           t,
            "device_id":           device_id,
            "device_type":         device_type,
            "attack_variant":      attack_variant if attack_variant is not None else "normal",
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

    variant_names = list(VARIANTS.keys())   # cycle order for attacked devices
    all_dfs = []

    for device_type in device_types:
        for i in range(n_devices_per_type):
            # Normal device
            dev_id = f"{device_type}_normal_{i}"
            df = generate_device_timeline(
                device_id=dev_id,
                device_type=device_type,
                n_timesteps=n_timesteps,
                attack_variant=None,
                attack_start=attack_start,
                rng=rng,
            )
            all_dfs.append(df)

            # Attacked device — cycle through variants
            variant = variant_names[i % len(variant_names)]
            dev_id  = f"{device_type}_attack_{variant}_{i}"
            df = generate_device_timeline(
                device_id=dev_id,
                device_type=device_type,
                n_timesteps=n_timesteps,
                attack_variant=variant,
                attack_start=attack_start,
                rng=rng,
            )
            all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    out_path = os.path.join(out_dir, "icu_simulation.csv")
    combined.to_csv(out_path, index=False)

    total    = len(combined)
    n_normal = int((combined["label"] == 0).sum())
    print(f"Simulation complete.")
    print(f"Total rows     : {total:,}")
    print(f"Normal rows    : {n_normal:,} ({100 * n_normal / total:.1f}%)")
    for name, lbl in VARIANTS.items():
        n = int((combined["label"] == lbl).sum())
        print(f"  label={lbl} ({name:20s}): {n:,} ({100 * n / total:.1f}%)")
    print(f"Devices        : {combined['device_id'].nunique()}")
    print(f"Saved to       : {out_path}")


if __name__ == "__main__":
    main()
