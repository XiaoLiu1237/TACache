"""Shared TACache runtime for model adapters.

Calibration records velocities; inference uses the table to predict skipped
steps. Model-specific input layouts and prediction details live in adapters.
"""
import copy
from contextlib import contextmanager
import functools
import json
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .utils import write_json
from .calibration import build_skip_timestep_from_calib_records, load_skip_table

TensorTree = List[torch.Tensor]
ExtractOutputFn = Callable[[Any], TensorTree]
MakeOutputFn = Callable[[TensorTree], Any]
ALGORITHM_VERSION = "tacache-unified-0.2"

@dataclass
class TACacheConfig:
    """Runtime knobs for wrapping a new diffusion/flow denoiser."""

    num_steps: int
    num_streams: int = 1
    method_name: str = "forward"
    output_prefix: str = "tacache"
    skip_path: Optional[str] = None
    tau_k: float = 0.03
    tau_theta: float = 0.30
    h_cap: int = 3
    lambda_std: float = 1.0
    min_h: int = 1
    eps: float = 1e-8
    timestep_scale: float = 1.0
    timestep_arg_names: Tuple[str, ...] = ("t", "timestep", "sigma")
    timestep_arg_positions: Tuple[int, ...] = (1,)
    verbose: bool = True

    # --- input-aware geometry ---
    cache_mode: str = "auto"   # "auto" | "global" | "image" | "video"
    t_blocks: int = 5          # temporal blocks for video mode
    temporal_axis: int = 2     # T axis in [B, C, T, H, W]
    sample_axis: int = 0       # batch axis for image per-sample mode

    # External CFG pairs cond/uncond forwards and caches the combined velocity:
    #     v = uncond + cfg_scale * (cond - uncond)
    # A skipped pair returns the same combined v to both branches, preserving
    # the native loop's combination. Methods that already return a combined
    # velocity use cfg_scale=None and num_streams=1. The default branch order
    # is cond, then uncond.
    cfg_scale: Optional[float] = None
    cfg_combine: Optional[Callable[[TensorTree, TensorTree, float], TensorTree]] = None

    num_train_timesteps: float = 1000.0
    sigma_fn: Optional[Callable[[float], float]] = None
    sigma_fn_id: Optional[str] = None  # stable identifier for a custom mapping
    # Automatic time selection compares the complete native and sigma sequences.
    time_coordinate: str = "auto"       # auto | native | sigma
    timestep_semantics: Optional[str] = None  # native_t | sigma | scaled_sigma
    video_batch_axis: Optional[int] = None   # None: auto for 5D; set 0 for packed H3
    model_family: Optional[str] = None
    workflow: Optional[str] = None
    calibration_context: Dict[str, Any] = field(default_factory=dict)
    require_table_metadata: bool = False
    scheduler: Any = field(default=None, repr=False, compare=False)


def resolve_time_coordinate(cfg):
    if cfg.time_coordinate not in ("auto", "native", "sigma"):
        raise ValueError("time_coordinate must be auto, native, or sigma")
    semantics = cfg.timestep_semantics
    if semantics not in (None, "native_t", "sigma", "scaled_sigma"):
        raise ValueError("Declare timestep_semantics as native_t, sigma, or scaled_sigma")
    if cfg.time_coordinate == "sigma" and semantics == "native_t" and cfg.sigma_fn is None:
        raise ValueError("native_t -> sigma requires an explicit scheduler mapping (sigma_fn)")
    if cfg.sigma_fn is not None and cfg.require_table_metadata and not cfg.sigma_fn_id:
        raise ValueError("A custom sigma_fn requires a stable sigma_fn_id for table validation")
    if cfg.time_coordinate != "auto":
        return cfg.time_coordinate
    if semantics in ("sigma", "scaled_sigma"):
        return "sigma"
    if semantics == "native_t":
        return "native"
    return "native"  # observe full calibration trajectory before auto selection


def table_metadata(cfg):
    return {"algorithm_version": ALGORITHM_VERSION, "calibration_config": {
        "model_family": cfg.model_family, "workflow": cfg.workflow,
        "num_steps": cfg.num_steps, "num_streams": cfg.num_streams,
        "method_name": cfg.method_name, "cache_mode": cfg.cache_mode,
        "t_blocks": cfg.t_blocks, "temporal_axis": cfg.temporal_axis,
        "sample_axis": cfg.sample_axis, "video_batch_axis": cfg.video_batch_axis,
        "time_coordinate": resolve_time_coordinate(cfg),
        "timestep_semantics": cfg.timestep_semantics,
        "timestep_scale": cfg.timestep_scale, "num_train_timesteps": cfg.num_train_timesteps,
        "cfg_scale": cfg.cfg_scale, "sigma_fn_id": cfg.sigma_fn_id,
        "context": copy.deepcopy(cfg.calibration_context)}}


def validate_table_metadata(obj, cfg):
    actual = obj.get("calibration_config")
    if actual is None:
        if cfg.require_table_metadata:
            raise ValueError("Legacy table has no model/time/layout metadata. Recalibrate with this delivery.")
        return
    if obj.get("algorithm_version") != ALGORITHM_VERSION:
        raise ValueError("Calibration algorithm version differs; recalibrate.")
    expected = table_metadata(cfg)["calibration_config"]
    differences = [key for key in expected if actual.get(key) != expected[key]]
    if differences:
        raise ValueError("Calibration table/config mismatch: " + ", ".join(differences))


# Time coordinates follow actual native timesteps and scheduler values.
@dataclass(frozen=True)
class TimeDecision:
    coordinate: str
    semantics: str
    scale: float = 1.0
    reason: str = ""


def decide_time_coordinate(observed_timesteps=None, scheduler=None, *, sigmas=None):
    """Automatically choose native t / sigma / scaled sigma from whole arrays.

    Equal sequences select sigma. A constant positive ratio selects scaled
    sigma. Other relationships use native t in the denoiser's units.
    """
    if callable(scheduler):
        scheduler = scheduler()
    noise = sigmas if sigmas is not None else getattr(scheduler, "sigmas", None)
    arrays = []
    for value in (observed_timesteps, noise):
        if value is None:
            value = []
        elif torch.is_tensor(value):
            value = value.detach().float().cpu().numpy()
        arrays.append(np.asarray(value, dtype=float).reshape(-1))
    times, noise = arrays
    if len(times) >= 2 and len(noise) in (len(times), len(times)+1):
        noise = noise[:len(times)]  # terminal sigma has no denoiser call
        if np.all(np.isfinite(times)) and np.all(np.isfinite(noise)):
            if np.allclose(times, noise, rtol=1e-4, atol=1e-6):
                return TimeDecision("sigma", "sigma", reason="actual forward times equal sigma")
            if np.allclose(times, 1.0-noise, rtol=1e-4, atol=1e-6):
                return TimeDecision("native", "native_t", reason="actual forward times equal 1-sigma; keep native t")
            mask = np.abs(noise) > 1e-8
            ratios = times[mask] / noise[mask]
            if len(ratios) >= 2:
                scale = float(np.median(ratios))
                if scale > 0 and np.allclose(times, scale*noise, rtol=1e-4, atol=1e-6):
                    return TimeDecision("sigma", "scaled_sigma", scale,
                                        f"complete actual trajectory proves t={scale:g}*sigma")
    return TimeDecision("native", "native_t", reason="no verified linear sigma mapping; use actual native t")


# Group outputs by image sample or video time block.


def _resolve_mode(tensors, cfg):
    mode = cfg.cache_mode
    if mode and mode != "auto":
        return mode
    max_dim = max((t.dim() for t in tensors), default=0)
    if max_dim >= 5:
        return "video"
    if max_dim == 4:
        return "image"
    return "global"


def _geometry_groups(x, mode, cfg, block_count=None):
    """Yield the image or temporal-block slices used by the geometry formulas."""
    if mode == "video":
        axis = cfg.temporal_axis
        n = int(x.shape[axis])
        groups = min(block_count if block_count is not None else cfg.t_blocks, n)
        bounds = torch.linspace(0, n, steps=groups + 1, dtype=torch.long).tolist()
        spans = zip(bounds[:-1], bounds[1:])
        batch_axis = cfg.video_batch_axis
        if batch_axis is None and x.ndim >= 5:
            batch_axis = 0
    elif mode == "image":
        axis = cfg.sample_axis
        spans = [(i, i + 1) for i in range(x.shape[axis])]
        batch_axis = None
    else:
        axis = 0
        spans = [(0, x.shape[0])]
        batch_axis = None
    if batch_axis == axis:
        raise ValueError("Video batch axis and temporal axis must be different")
    spans = list(spans)
    for b in range(x.shape[batch_axis] if batch_axis is not None else 1):
        for block, (start, end) in enumerate(spans):
            index = [slice(None)] * x.ndim
            index[axis] = slice(start, end)
            if batch_axis is not None:
                index[batch_axis] = slice(b, b + 1)
            yield block, tuple(index)


def _grouped_perp(x_list, ref_list, mode, cfg):
    """Per-group orthogonalize x against ref and normalize within each group."""
    eps = cfg.eps
    out_list = []
    blocks = min(cfg.t_blocks, min(x.shape[cfg.temporal_axis] for x in x_list)) if mode == "video" else None
    for x, ref in zip(x_list, ref_list):
        x_f = x.to(torch.float32)
        ref_f = ref.to(torch.float32)
        out = torch.zeros_like(x_f)
        for _block, index in _geometry_groups(x_f, mode, cfg, blocks):
            x_blk = x_f[index]
            r_blk = ref_f[index]
            x_flat = x_blk.reshape(-1)
            r_flat = r_blk.reshape(-1)
            vv = (r_flat * r_flat).sum().clamp_min(eps)
            coeff = (x_flat * r_flat).sum() / vv
            x_perp = x_blk - coeff * r_blk
            norm = x_perp.reshape(-1).norm().clamp_min(eps)
            out[index] = x_perp / norm
        out_list.append(out)
    return out_list


def _group_calibration_stats(cur_list, prev_list, dt_pair, mode, cfg):
    """Per-group (k, k0, r_norm, alpha) for the *video* per-block curves.

    Returns four lists of length ``t_blocks`` (block-index aligned across the
    output tensors by even temporal split). Image/global modes don't use this.
    """
    eps = cfg.eps
    width = min(cfg.t_blocks, min(x.shape[cfg.temporal_axis] for x in cur_list))
    k_blocks, k0_blocks, r_blocks, alpha_blocks = [], [], [], []
    for blk in range(width):
        cur_parts, prev_parts = [], []
        for cur, prev in zip(cur_list, prev_list):
            axis = cfg.temporal_axis
            n = int(cur.shape[axis])
            bounds = torch.linspace(0, n, steps=width + 1, dtype=torch.long).tolist()
            t0, t1 = int(bounds[blk]), int(bounds[blk + 1])
            index = [slice(None)] * cur.ndim
            index[axis] = slice(t0, t1)
            cur_parts.append(cur.to(torch.float32)[tuple(index)].reshape(-1))
            prev_parts.append(prev.to(torch.float32)[tuple(index)].reshape(-1))
        cur_flat = torch.cat(cur_parts)
        prev_flat = torch.cat(prev_parts)
        v_norm = cur_flat.norm()
        vp_norm = prev_flat.norm()
        a_hat = (cur_flat - prev_flat) / dt_pair
        k = (torch.log(v_norm + eps) - torch.log(vp_norm + eps)) / dt_pair
        dot = (a_hat * prev_flat).sum()
        v_sq = (prev_flat * prev_flat).sum() + eps
        k0 = dot / v_sq
        r = a_hat - k0 * prev_flat
        r_norm = r.norm()
        alpha = (r_norm * dt_pair) / (vp_norm + eps)
        k_blocks.append(float(k.item()))
        k0_blocks.append(float(k0.item()))
        r_blocks.append(float(r_norm.item()))
        alpha_blocks.append(float(alpha.item()))
    return k_blocks, k0_blocks, r_blocks, alpha_blocks


class TACacheWrapper:
    """Track calibration/CFG/cache state around one original model method.

    Geometry operates on lists of tensors. Model adapters provide extract/make
    callbacks; enable_tacache handles native Tensor/List/Tuple outputs at entry.
    """

    # Adapters can preserve their native reduction order and precision.
    _grouped_perp = staticmethod(_grouped_perp)

    def __init__(
        self,
        model,
        config: TACacheConfig,
        extract_output: Optional[ExtractOutputFn] = None,
        make_output: Optional[MakeOutputFn] = None,
    ):
        if config.num_steps <= 0:
            raise ValueError("TACacheConfig.num_steps must be positive.")
        if config.num_streams <= 0:
            raise ValueError("TACacheConfig.num_streams must be positive.")
        if config.t_blocks < 1:
            raise ValueError("t_blocks must be positive")
        if config.num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")
        resolve_time_coordinate(config)

        self.model = model
        self.config = copy.copy(config)
        self.steps_per_stream = config.num_steps // config.num_streams
        self._time_ready = False
        self.time_decision = None
        self._initial_time = (config.time_coordinate, config.timestep_semantics, config.num_train_timesteps)
        self.extract_output = extract_output or (lambda output: output)
        self.make_output = make_output or (lambda sample: sample)
        self.original_method = getattr(model, config.method_name)
        self._loaded_table_object = {}
        self.mode = "disabled"
        self.cnt = 0
        self.tables = None
        self.last_calibration_result = None
        self.last_skip_count = 0
        self.reset_calibration_records()
        self.reset_runtime(clear_counts=True)


    def restore(self):
        setattr(self.model, self.config.method_name, self.original_method)
        self.mode = "disabled"

    def reset_runtime(self, clear_counts=False):
        n = self.config.num_streams
        self.v_state = [None] * n           # list[list[Tensor]] per stream
        self.u_state = [None] * n
        self.geo_mode = [None] * n
        self.skip_remaining = [0] * n
        if clear_counts:
            self.skip_counts = [0] * n
        # external-CFG pairing buffers (used only when config.cfg_scale is set)
        self._cfg_buf0 = None        # branch-0 (cond) output, awaiting branch-1
        self._cfg_skip_step = False  # is the current paired step a skip?
        self._cfg_skip_out = None    # combined v to return on both branches

    def reset_calibration_records(self):
        n = self.config.num_streams
        # one trajectory record per stream (the calib-record schema)
        self.calib_records = [{
            "t": [], "dt_update": [], "dt_pair": [], "v_norm": [],
            "k": [], "k_0": [], "r_norm": [],
            "blk_k": [], "blk_rdt": [],   # per-block, video mode only
        } for _ in range(n)]
        self.calib_prev_outputs = [None] * n
        self.calib_prev_flat = [None] * n
        self.calib_prev_t = [None] * n

    def _reset_time_selection(self):
        self._time_ready = False
        self.time_decision = None
        (self.config.time_coordinate, self.config.timestep_semantics,
         self.config.num_train_timesteps) = self._initial_time

    def install_calibration(self):
        self._reset_time_selection()
        wrapper = self

        def wrapped(*args, **kwargs):
            wrapper._prepare_call(args, kwargs)
            return wrapper._calibration_forward(*args, **kwargs)

        # Preserve the original forward signature for model integrations.
        functools.update_wrapper(wrapped, self.original_method)
        setattr(self.model, self.config.method_name, wrapped)
        self.mode = "calibration"
        self.cnt = 0
        self.last_calibration_result = None
        self.reset_runtime(clear_counts=True)
        self.reset_calibration_records()

    def install_inference(self, skip_path=None):
        self._reset_time_selection()
        with open(skip_path or self.config.skip_path, encoding="utf-8") as handle:
            self._loaded_table_object = json.load(handle)
        self.tables = load_skip_table(
            skip_path or self.config.skip_path,
            num_streams=self.config.num_streams,
            sigma_fn=self._effective_sigma_fn(),
        )
        if any(len(table["h_max_off"]) != self.steps_per_stream - 1 for table in self.tables):
            raise ValueError("Skip table length differs from the configured sampling steps")
        wrapper = self

        def wrapped(*args, **kwargs):
            wrapper._prepare_call(args, kwargs)
            return wrapper._inference_forward(*args, **kwargs)

        functools.update_wrapper(wrapped, self.original_method)
        setattr(self.model, self.config.method_name, wrapped)
        self.mode = "inference"
        self.cnt = 0
        self.last_skip_count = 0
        self.reset_runtime(clear_counts=True)

    def _prepare_call(self, args, kwargs):
        """Model adapters can inspect inputs before each forward call."""
        return None

    def _decide_time(self):
        # Calibration always measures in native units first. Full-trajectory
        # numeric evidence is available at finalize, before writing a table.
        if self.mode != "inference" or self._time_ready:
            return
        cfg = self.config
        obj = self._loaded_table_object
        stored = obj.get("time_decision")
        if cfg.time_coordinate == "auto" and cfg.timestep_semantics is None:
            if stored:
                decision = TimeDecision(**stored)
                if cfg.scheduler is not None:
                    live = decide_time_coordinate(obj["full_timesteps"], cfg.scheduler)
                    if (live.coordinate, live.semantics) != (decision.coordinate, decision.semantics) or not np.isclose(live.scale, decision.scale):
                        raise ValueError("Runtime sigma/time relationship differs from calibration; recalibrate")
            else:
                decision = decide_time_coordinate()
            cfg.time_coordinate = decision.coordinate
            cfg.timestep_semantics = decision.semantics
            cfg.num_train_timesteps = decision.scale
            self.time_decision = decision
        validate_table_metadata(obj, cfg)
        self._time_ready = True

    def _validate_runtime_time(self, value, step_idx):
        self._decide_time()
        grid = self._loaded_table_object.get("full_timesteps")
        if grid is not None and (step_idx >= len(grid) or not np.isclose(value, grid[step_idx], rtol=1e-4, atol=1e-6)):
            raise ValueError(f"Actual forward timestep differs from calibration at step {step_idx}; recalibrate")

    def _finalize_time(self):
        cfg = self.config
        if cfg.time_coordinate != "auto" or cfg.timestep_semantics is not None:
            return
        valid = [r for r in self.calib_records if len(r["t"]) >= 2]
        decision = decide_time_coordinate(valid[0]["t"] if valid else None, cfg.scheduler)
        # The measured coefficients use native dt. For t=c*sigma, dt/c and
        # k*c leave k*dt invariant; alpha=r*dt/||v_prev|| is also invariant.
        scale = decision.scale if decision.coordinate == "sigma" else 1.0
        for record in valid:
            for key in ("dt_update", "dt_pair"):
                record[key] = [None if v is None else v/scale for v in record[key]]
            for key in ("k", "k_0", "r_norm"):
                record[key] = [None if v is None else v*scale for v in record[key]]
            record["blk_k"] = [None if row is None else [None if v is None else v*scale for v in row]
                                 for row in record["blk_k"]]
        cfg.time_coordinate = decision.coordinate
        cfg.timestep_semantics = decision.semantics
        cfg.num_train_timesteps = decision.scale
        self.time_decision = decision
        if cfg.verbose:
            print(f"[tacache time] {decision.coordinate}: {decision.reason}")

    def _extract_timestep(self, args, kwargs):
        for name in self.config.timestep_arg_names:
            if name in kwargs:
                return self._timestep_value(kwargs[name])
        for position in self.config.timestep_arg_positions:
            if len(args) > position:
                return self._timestep_value(args[position])
        raise TypeError("Denoiser call has no timestep argument")

    def _timestep_value(self, timestep):
        if torch.is_tensor(timestep):
            value = timestep.detach().float().mean().item()
        elif isinstance(timestep, (list, tuple)):
            values = [
                item.detach().float().mean().item() if torch.is_tensor(item) else float(item)
                for item in timestep
            ]
            value = float(np.mean(values))
        else:
            value = float(timestep)

        scale = self.config.timestep_scale
        if scale is not None and scale > 0 and scale != 1.0:
            value = value / scale
        return value

    def _get_h(self, stream, step_idx):
        table = self.tables[stream]
        total = self.steps_per_stream
        if step_idx in (0, 1, total - 2, total - 1):
            return 1
        h = int(table["h_max_off"][step_idx])
        h = max(int(self.config.min_h), h)
        h = min(h, (total - 2) - step_idx)
        return max(1, h)

    def _can_skip(self, stream, step_idx):
        total = self.steps_per_stream
        return (
            step_idx not in (0, 1, total - 2, total - 1)
            and self.skip_remaining[stream] > 0
            and self.v_state[stream] is not None
            and self.u_state[stream] is not None
        )

    def _effective_sigma_fn(self):
        # Explicit override or verified numeric decision; native t otherwise.
        if self.config.sigma_fn is not None:
            return self.config.sigma_fn
        if resolve_time_coordinate(self.config) == "sigma":
            if self.config.timestep_semantics == "sigma":
                return lambda t: float(t)
            if self.config.timestep_semantics == "native_t":
                raise ValueError("native_t -> sigma needs an explicit sigma_fn")
            ntt = float(self.config.num_train_timesteps)
            return lambda t: float(t) / ntt
        return None

    def _sigma(self, t):
        # Express time in the selected coordinate: mapped sigma or native t.
        sf = self._effective_sigma_fn()
        return sf(t) if sf is not None else float(t)

    def _advance_tacache_state(self, stream, step_idx):
        transition_idx = step_idx - 1
        table = self.tables[stream]
        k_row = table["k_off_used"][transition_idx]
        alpha_row = table["r_dt_over_v_off_used"][transition_idx]
        k_is_2d = np.ndim(k_row) >= 1
        k_vec = np.atleast_1d(k_row).astype(np.float32)
        alpha_vec = np.atleast_1d(alpha_row).astype(np.float32)
        eps = self.config.eps
        mode = self.geo_mode[stream]
        v_state = self.v_state[stream]
        u_state = self.u_state[stream]
        dt_j = max(float(table["dt_update_used"][transition_idx]), eps)
        u_perp = self._grouped_perp(u_state, v_state, mode, self.config)  # re-orth to current v

        next_outputs = []
        blocks = min(self.config.t_blocks, min(x.shape[self.config.temporal_axis] for x in v_state)) if mode == "video" else None
        if k_is_2d and blocks is not None and len(k_vec) != blocks:
            raise ValueError("Video block count differs from calibration; recalibrate")
        for v_cur, u_cur in zip(v_state, u_perp):
            v_f = v_cur.to(torch.float32)
            u_f = u_cur.to(torch.float32)
            out = torch.zeros_like(v_f)
            for g, index in _geometry_groups(v_f, mode, self.config, blocks):
                # per-block curve for video (2D table); scalar shared otherwise
                k_g = float(k_vec[min(g, len(k_vec) - 1)]) if k_is_2d else float(k_vec[0])
                a_g = float(alpha_vec[min(g, len(alpha_vec) - 1)]) if k_is_2d else float(alpha_vec[0])
                v_blk = v_f[index]
                u_blk = u_f[index]
                vnorm = v_blk.reshape(-1).norm().clamp_min(eps)
                scale = torch.exp(torch.tensor(
                    k_g * dt_j, device=v_f.device, dtype=torch.float32))
                out[index] = v_blk * scale + a_g * vnorm * u_blk
            next_outputs.append(out.to(device=v_cur.device, dtype=v_cur.dtype))

        self.v_state[stream] = [u.detach() for u in next_outputs]
        self.skip_remaining[stream] -= 1
        self.skip_counts[stream] += 1
        return next_outputs

    def _after_actual_forward(self, sample, stream, step_idx):
        current = [t.detach() for t in sample]
        mode = _resolve_mode(current, self.config)
        self.geo_mode[stream] = mode
        prev = self.v_state[stream]
        self.skip_remaining[stream] = 0

        if prev is not None:
            total = self.steps_per_stream
            if step_idx not in (0, 1, total - 2, total - 1):
                delta = [c - p for c, p in zip(current, prev)]
                # u_state = orthonormal update direction (delta ⟂ prev), per group
                self.u_state[stream] = self._grouped_perp(delta, prev, mode, self.config)
                h = self._get_h(stream, step_idx)
                self.skip_remaining[stream] = max(0, h - 1)

        self.v_state[stream] = current

    def _record_calibration(self, sample, stream, step_idx, t_cur):
        record = self.calib_records[stream]
        current = [t.detach() for t in sample]
        mode = _resolve_mode(current, self.config)
        flat = torch.cat([t.reshape(-1) for t in current], dim=0).detach().to(torch.float32)
        v_norm = flat.norm()
        prev_outputs = self.calib_prev_outputs[stream]
        prev_flat = self.calib_prev_flat[stream]
        prev_t = self.calib_prev_t[stream]
        eps = self.config.eps

        if record["dt_update"] and record["dt_update"][-1] is None and prev_t is not None:
            record["dt_update"][-1] = max(abs(self._sigma(prev_t) - self._sigma(t_cur)), eps)

        record["t"].append(float(t_cur))
        record["dt_update"].append(None)
        record["v_norm"].append(float(v_norm.item()))

        if prev_flat is None or prev_t is None or prev_outputs is None:
            record["dt_pair"].append(None)
            record["k"].append(None)
            record["k_0"].append(None)
            record["r_norm"].append(None)
            record["blk_k"].append(None)
            record["blk_rdt"].append(None)
        else:
            dt_pair = max(abs(self._sigma(prev_t) - self._sigma(t_cur)), eps)
            vp_norm = prev_flat.norm()
            k = (torch.log(v_norm + eps) - torch.log(vp_norm + eps)) / dt_pair
            a_hat = (flat - prev_flat) / dt_pair
            dot = (a_hat * prev_flat).sum()
            v_sq = (prev_flat * prev_flat).sum() + eps
            k_0 = dot / v_sq
            r = a_hat - k_0 * prev_flat
            r_norm = r.norm()

            record["dt_pair"].append(float(dt_pair))
            record["k"].append(float(k.item()))
            record["k_0"].append(float(k_0.item()))
            record["r_norm"].append(float(r_norm.item()))

            if mode == "video":
                k_blk, _k0_blk, _r_blk, alpha_blk = _group_calibration_stats(
                    current, prev_outputs, dt_pair, mode, self.config)
                record["blk_k"].append(k_blk)
                record["blk_rdt"].append(alpha_blk)
            else:
                record["blk_k"].append(None)
                record["blk_rdt"].append(None)

        self.calib_prev_outputs[stream] = current
        self.calib_prev_flat[stream] = flat
        self.calib_prev_t[stream] = t_cur

    def _combine_cfg(self, branch0, branch1):
        scale = self.config.cfg_scale
        if self.config.cfg_combine is not None:
            return self.config.cfg_combine(branch0, branch1, scale)
        return [u + scale * (c - u) for c, u in zip(branch0, branch1)]

    def _calibration_forward(self, *args, **kwargs):
        step_idx, stream = divmod(self.cnt, self.config.num_streams)
        t_cur = self._extract_timestep(args, kwargs)
        output = self.original_method(*args, **kwargs)
        sample = self.extract_output(output)
        if self.config.cfg_scale is None:
            self._record_calibration(sample, stream, step_idx, t_cur)
        elif stream == 0:
            self._cfg_buf0 = sample
        else:
            combined = self._combine_cfg(self._cfg_buf0, sample)
            self._record_calibration(combined, 0, step_idx, t_cur)
            self._cfg_buf0 = None
        self._finish_forward_step()
        return output

    def _inference_forward(self, *args, **kwargs):
        step_idx, stream = divmod(self.cnt, self.config.num_streams)
        t_cur = self._extract_timestep(args, kwargs)
        self._validate_runtime_time(t_cur, step_idx)

        if self.config.cfg_scale is None:
            if self._can_skip(stream, step_idx):
                output = self.make_output(self._advance_tacache_state(stream, step_idx))
            else:
                output = self.original_method(*args, **kwargs)
                self._after_actual_forward(self.extract_output(output), stream, step_idx)
        elif stream == 0:
            # CFG branches share one skip decision and one cached velocity.
            self._cfg_skip_step = self._can_skip(0, step_idx)
            if self._cfg_skip_step:
                self._cfg_skip_out = self._advance_tacache_state(0, step_idx)
                output = self.make_output(self._cfg_skip_out)
            else:
                output = self.original_method(*args, **kwargs)
                self._cfg_buf0 = self.extract_output(output)
        elif self._cfg_skip_step:
            output = self.make_output(self._cfg_skip_out)
        else:
            output = self.original_method(*args, **kwargs)
            combined = self._combine_cfg(self._cfg_buf0, self.extract_output(output))
            self._after_actual_forward(combined, 0, step_idx)
            self._cfg_buf0 = None

        self._finish_forward_step()
        return output

    def _finish_forward_step(self):
        self.cnt += 1
        if self.cnt < self.config.num_steps:
            return

        if self.mode == "calibration":
            self.finalize_calibration(reset=True)
        elif self.mode == "inference":
            self.last_skip_count = int(sum(self.skip_counts))
            if self.config.verbose:
                print(f"TACache skipped forwards: {self.skip_counts}")
            self.cnt = 0
            self.reset_runtime(clear_counts=True)

    def finalize_calibration(self, reset=False):
        self._finalize_time()
        records = [copy.deepcopy(record) for record in self.calib_records]
        valid_records = [record for record in records if len(record.get("t", [])) >= 2]
        if not valid_records:
            raise RuntimeError("Calibration has no complete velocity records")

        table_kwargs = dict(
            tau_k=self.config.tau_k,
            tau_theta=self.config.tau_theta,
            h_cap=self.config.h_cap,
            lambda_std=self.config.lambda_std,
            eps=self.config.eps,
        )

        stream_tables = []
        for stream, record in enumerate(records):
            if len(record["t"]) < 2:
                continue
            table = build_skip_timestep_from_calib_records([record], **table_kwargs)
            table["stream"] = stream
            stream_tables.append(table)

        combined_table = build_skip_timestep_from_calib_records(
            valid_records, **table_kwargs)
        calib_path = f"{self.config.output_prefix}_calib_records.json"
        skip_path = f"{self.config.output_prefix}_skip_timestep.json"

        write_json(
            calib_path,
            {
                "format": "tacache_calib_records_v1",
                "num_streams": self.config.num_streams,
                "records": [
                    {"stream": stream, "calib_recorder": record}
                    for stream, record in enumerate(records)
                ],
            },
        )

        skip_obj = {
            "format": "tacache_skip_table_v1",
            "num_streams": self.config.num_streams,
            "tables": stream_tables,
            "combined": combined_table,
        }
        skip_obj.update(combined_table)
        skip_obj.update(table_metadata(self.config))
        skip_obj["full_timesteps"] = valid_records[0]["t"]
        if self.time_decision is not None:
            skip_obj["time_decision"] = dict(self.time_decision.__dict__)
        write_json(skip_path, skip_obj)

        if self.config.verbose:
            print(f"TACache calibration saved to {calib_path} and {skip_path}")

        self.last_calibration_result = {
            "calib_path": calib_path,
            "skip_path": skip_path,
            "skip_table": skip_obj,
        }

        if reset:
            self.cnt = 0
            self.reset_runtime(clear_counts=True)
            self.reset_calibration_records()
        return self.last_calibration_result


# Public context manager for one complete native generation.
def layout_fields(layout=None):
    """Select geometry from the real B/T axes of unpacked output."""
    if layout is None:
        return dict(cache_mode="auto")
    if not layout.isalpha() or layout != layout.upper() or len(set(layout)) != len(layout):
        raise ValueError("layout must have unique axis letters, e.g. BCHW or BCTHW")
    if "T" in layout:
        return dict(cache_mode="video", temporal_axis=layout.index("T"),
                    video_batch_axis=layout.index("B") if "B" in layout else None)
    return dict(cache_mode="image" if "B" in layout else "global",
                sample_axis=layout.index("B") if "B" in layout else 0)


@contextmanager
def enable_tacache(model, mode, config, *, skip_table=None, layout=None,
                   extract_output=None, make_output=None, wrapper_cls=TACacheWrapper):
    """Attach TACache for one generation and restore forward even on failure."""
    if mode not in ("baseline", "calibrate", "infer"):
        raise ValueError("mode must be baseline, calibrate or infer")
    if mode == "infer" and not skip_table:
        raise ValueError("infer requires --skip-table")
    if config.num_streams < 1 or config.num_steps < 2 or config.num_steps % config.num_streams:
        raise ValueError("num_steps must contain complete denoiser streams")
    if config.cfg_scale is not None and config.num_streams != 2:
        raise ValueError("External CFG requires two alternating streams")
    context = {**config.calibration_context, "adapter_layout": layout}
    config = replace(config, **layout_fields(layout), calibration_context=context)
    cache = wrapper_cls(model, config, extract_output, make_output)
    cache.calibration, cache.report = None, None
    cache.sampler_calls = 0
    geometry, calls = None, 0
    extract = cache.extract_output

    def checked_extract(output):
        nonlocal geometry
        sample = extract(output)
        tensor_output = torch.is_tensor(sample)
        tensors = [sample] if tensor_output else list(sample)
        if make_output is None:
            cache.make_output = (lambda values: values[0]) if tensor_output else type(sample)
        if layout and any(t.ndim != len(layout) for t in tensors):
            raise ValueError("Denoiser output rank differs from its real layout")
        current = _resolve_mode(tensors, config)
        if geometry is not None and geometry != current:
            raise ValueError("Output geometry changed within a generation")
        expected = cache._loaded_table_object.get("resolved_geometry")
        if expected is not None and expected != current:
            raise ValueError("Actual geometry differs from calibration")
        geometry = current
        return tensors

    try:
        if mode != "baseline":
            cache.extract_output = checked_extract
            if mode == "calibrate":
                cache.install_calibration()
            else:
                cache.install_inference(skip_table)
        active = getattr(model, config.method_name)

        @functools.wraps(active)
        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return active(*args, **kwargs)

        setattr(model, config.method_name, counted)
        yield cache
        # Native samplers can advance cached steps without entering forward.
        calls += cache.sampler_calls
        if calls != config.num_steps or cache.cnt:
            raise RuntimeError(f"Expected {config.num_steps} denoiser calls, observed {calls}")
        cache.report = dict(denoiser_calls=calls, geometry=geometry,
            skip_count=cache.last_skip_count or sum(cache.skip_counts),
            time_decision=dict(cache.time_decision.__dict__) if cache.time_decision else None)
        if mode == "calibrate":
            result = cache.last_calibration_result
            table = result["skip_table"]
            metadata = {key: table[key] for key in
                        ("algorithm_version", "calibration_config", "time_decision", "full_timesteps")}
            metadata["resolved_geometry"] = geometry
            records = json.loads(Path(result["calib_path"]).read_text(encoding="utf-8"))["records"]
            cache.calibration = dict(metadata=metadata, records=records)
    finally:
        cache.restore()
