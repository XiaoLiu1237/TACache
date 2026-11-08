"""Calibration records -> conservative curves -> skip lengths.

build_skip_timestep_from_calib_records is the one table builder for the models.
save_calibration aggregates completed generations from a prompt set.
"""
from __future__ import annotations
import json
from pathlib import Path
import warnings
import numpy as np

from .utils import write_json

def _build_hmax_table(dt_update, k_off, r_off, tau_k, tau_theta, h_cap):
    n = int(len(dt_update))
    h_k = np.ones(n, dtype=np.int32)
    h_r = np.ones(n, dtype=np.int32)
    h_max = np.ones(n, dtype=np.int32)

    for i in range(n):
        h_lim = min(h_cap, n - i)
        sum_k = 0.0
        count_h = 0
        for j in range(i, i + h_lim):
            v = k_off[j]
            dt = dt_update[j]
            if np.isnan(v) or np.isnan(dt):
                break
            sum_k += abs(v) * abs(dt)
            if sum_k > tau_k:
                break
            count_h += 1
        h_k[i] = max(1, count_h)

    for i in range(n):
        h_lim = min(h_cap, n - i)
        sum_r = 0.0
        count_r = 0
        for j in range(i, i + h_lim):
            v = r_off[j]
            if np.isnan(v):
                break
            sum_r += v
            if sum_r > tau_theta:
                break
            count_r += 1
        h_r[i] = max(1, count_r)

    for i in range(n):
        h_max[i] = max(1, min(h_k[i], h_r[i], h_cap))

    return h_k, h_r, h_max


def build_skip_timestep_from_calib_records(
    calib_records,
    tau_k=0.03,
    tau_theta=0.30,
    h_cap=3,
    lambda_std=1.0,
    eps=1e-8,
):
    # --- calibration table helpers ---
    def _json_float(x):
        if x is None:
            return None
        x = float(x)
        if not np.isfinite(x):
            raise ValueError("Calibration produced non-finite statistics")
        return x

    def _to_np_float_list(values):
        return np.asarray([np.nan if v is None else float(v) for v in values],
                          dtype=np.float64)

    def _to_np_float_matrix(rows, width):
        # The first velocity has no measured change, so its row is None.
        out = np.full((len(rows), width), np.nan, dtype=np.float64)
        for i, row in enumerate(rows):
            if row is None:
                continue
            out[i] = np.asarray(row, dtype=np.float64).reshape(width)
        return out

    records = []
    for item in calib_records:
        record = item["calib_recorder"] if "calib_recorder" in item else item
        if len(record["t"]) < 2:
            raise ValueError("Calibration records need at least two velocities")
        records.append(record)
    if not records:
        raise ValueError("No calibration records")

    for record in records:   # backfill last dt (warm-up steps leave it None)
        if record["dt_update"] and record["dt_update"][-1] is None:
            record["dt_update"][-1] = record["dt_pair"][-1]

    min_len = len(records[0]["t"])
    if any(len(record["t"]) != min_len for record in records):
        raise ValueError("Calibration trajectories have different lengths")
    t_arr = np.stack(
        [_to_np_float_list(record["t"][:min_len]) for record in records],
        axis=0,
    )
    dt_arr = np.stack(
        [_to_np_float_list(record["dt_update"][:min_len]) for record in records],
        axis=0,
    )
    # Curves have shape [records, steps] or [records, steps, time blocks].
    # The same aggregation, storage fields and reader handle both shapes.
    block_width = max(
        max((len(r) for r in rec["blk_k"] if r is not None), default=0)
        for rec in records
    )
    if block_width > 1:
        k_arr = np.stack(
            [_to_np_float_matrix(record["blk_k"][:min_len], block_width)
             for record in records],
            axis=0,
        )  # (R, min_len, W)
        rdt_arr = np.stack(
            [_to_np_float_matrix(record["blk_rdt"][:min_len], block_width)
             for record in records],
            axis=0,
        )
    else:
        k_arr = np.stack(
            [_to_np_float_list(record["k"][:min_len]) for record in records], axis=0,
        )
        rdt_rows = []
        for record in records:
            r_norm = _to_np_float_list(record["r_norm"][:min_len])
            dt_pair = _to_np_float_list(record["dt_pair"][:min_len])
            v_norm = _to_np_float_list(record["v_norm"][:min_len])
            v_prev_norm = np.full_like(v_norm, np.nan)
            v_prev_norm[1:] = v_norm[:-1]
            rdt_rows.append((r_norm * dt_pair) / (v_prev_norm + eps))
        rdt_arr = np.stack(rdt_rows, axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.nanmean(t_arr, axis=0)
            dt_mean = np.nanmean(dt_arr, axis=0)
            dt_std = np.nanstd(dt_arr, axis=0)
            k_mean = np.nanmean(k_arr, axis=0)
            k_std = np.nanstd(k_arr, axis=0)
            r_mean = np.nanmean(rdt_arr, axis=0)
            r_std = np.nanstd(rdt_arr, axis=0)

    # conservative curve = mean + lambda_std * std
    dt_update = dt_mean + lambda_std * dt_std
    k_off = k_mean + lambda_std * k_std
    r_off = r_mean + lambda_std * r_std

    n = min(len(dt_update), max(0, len(k_off) - 1), max(0, len(r_off) - 1))
    dt_use = dt_update[:n]
    k_use = k_off[1:1 + n]
    r_use = r_off[1:1 + n]

    # SSC uses the block mean; inference retains every block's own curve.
    if k_use.ndim == 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            with np.errstate(invalid="ignore", divide="ignore"):
                h_k_off = np.nanmean(k_use, axis=1)
                h_r_off = np.nanmean(r_use, axis=1)
    else:
        h_k_off = k_use
        h_r_off = r_use

    h_k, h_r, h_max = _build_hmax_table(
        dt_update=dt_use,
        k_off=h_k_off,
        r_off=h_r_off,
        tau_k=tau_k,
        tau_theta=tau_theta,
        h_cap=h_cap,
    )

    if not (np.isfinite(k_use).all() and np.isfinite(r_use).all()):
        raise ValueError("Calibration produced non-finite statistics")
    table = {
        "format": "tacache_skip_table_v1",
        "tau_k": tau_k,
        "tau_theta": tau_theta,
        "h_cap": h_cap,
        "lambda_std": lambda_std,
        "N": int(n),
        "num_calib_records": int(len(records)),
        "t": [_json_float(x) for x in t[:n].tolist()],
        "dt_update_used": [_json_float(x) for x in dt_use.tolist()],
        "k_off_used": k_use.tolist(),
        "r_dt_over_v_off_used": r_use.tolist(),
        "h_k": h_k.astype(int).tolist(),
        "h_r": h_r.astype(int).tolist(),
        "h_max_off": h_max.astype(int).tolist(),
        "summary": {
            "h_max_min": int(np.min(h_max)) if n > 0 else 0,
            "h_max_max": int(np.max(h_max)) if n > 0 else 0,
            "h_max_mean": float(np.mean(h_max)) if n > 0 else 0.0,
        },
    }

    if k_use.ndim == 2:
        table["block_count"] = int(block_width)

    return table


def load_skip_table(skip_path, num_streams=1, sigma_fn=None):
    """Load shared or per-stream skip curves and positive time intervals.

    Time intervals are read from dt_update_used or calculated from t.
    """
    with open(skip_path, encoding="utf-8") as handle:
        obj = json.load(handle)
    source_tables = obj["tables"] if "tables" in obj else [obj]
    if len(source_tables) not in (1, num_streams):
        raise ValueError("Skip table must contain one shared table or one per stream")
    tables = []
    for source in source_tables:
        k = np.asarray(source["k_off_used"], dtype=np.float32)
        direction = np.asarray(source["r_dt_over_v_off_used"], dtype=np.float32)
        h = np.asarray(source["h_max_off"], dtype=np.int64)
        if "dt_update_used" in source:
            dt = np.asarray(source["dt_update_used"], dtype=np.float32)
        else:
            times = source["t"]
            if sigma_fn is not None:
                times = [sigma_fn(t) for t in times]
            dt = np.abs(np.diff(times)).tolist()
            dt.append(dt[-1])
            dt = np.asarray(dt, dtype=np.float32)
        if h.ndim != 1 or k.ndim not in (1, 2) or k.shape != direction.shape or k.shape[0] != len(h) or dt.shape != h.shape:
            raise ValueError("Skip table curve shapes do not match")
        if "block_count" in source and k.shape != (len(h), source["block_count"]):
            raise ValueError("Time-block curves must have shape [steps, block_count]")
        if not (np.isfinite(k).all() and np.isfinite(direction).all() and np.isfinite(dt).all()):
            raise ValueError("Skip table contains non-finite calibration values")
        if np.any(dt <= 0) or np.any(h < 1):
            raise ValueError("Skip table requires positive time deltas and interval lengths")
        tables.append(dict(h_max_off=h, k_off_used=k, r_dt_over_v_off_used=direction, dt_update_used=dt))
    return tables * num_streams if len(tables) == 1 else tables


def save_calibration(calibrations, outdir, config):
    """Aggregate completed native generations with the one shared core builder."""
    if not calibrations:
        raise ValueError("No complete calibration generations")
    metadata = calibrations[0]["metadata"]
    records = []
    for index, item in enumerate(calibrations):
        if item["metadata"] != metadata:
            raise ValueError("Cannot combine different time grids, geometry or generation settings")
        # External CFG records only the combined velocity; the unused branch
        # has no trajectory. Nonempty but incomplete records still raise.
        records.extend({**record, "prompt_index":index} for record in item["records"]
                       if record["calib_recorder"]["t"])
    table = build_skip_timestep_from_calib_records(records, tau_k=config.tau_k,
        tau_theta=config.tau_theta, h_cap=config.h_cap, lambda_std=config.lambda_std, eps=config.eps)
    table.update(metadata, num_calib_prompts=len(calibrations), num_calib_records=len(records))
    outdir = Path(outdir)
    write_json(outdir / "tacache_calib_records.json", dict(records=records, **metadata))
    path = outdir / "tacache_skip_timestep.json"
    write_json(path, table)
    return str(path)
