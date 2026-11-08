"""Self-contained CPU tests for the TACache core.

Covers mode routing, grouped geometry, calibration/skip-table round trips,
external-CFG caching, and automatic time selection independent of output geometry.
"""

import os
import sys
import json
import shutil
import tempfile
import traceback

import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from tacache import core as tacache  # noqa: E402
from helpers import calibrate, infer

torch.manual_seed(0)
np.random.seed(0)

_RESULTS = []


def check(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
        _RESULTS.append(True)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {name}: {exc}")
        traceback.print_exc()
        _RESULTS.append(False)


# --------------------------------------------------------------------------- #
# synthetic calibration records
# --------------------------------------------------------------------------- #
def _new_rec():
    # the calib-record schema (TACache core inlined _new_calib_record into the wrapper)
    return {"t": [], "dt_update": [], "dt_pair": [], "v_norm": [],
            "k": [], "k_0": [], "r_norm": [], "blk_k": [], "blk_rdt": []}


def make_video_record(seed, T=12, B=5, global_k=1e-3, blk_k=0.05, blk_alpha=0.5):
    """Per-block (video) record. global_k and blk_k are set independently so a
    test can tell whether h is driven by the global scalar or the block mean."""
    rng = np.random.RandomState(seed)
    r = _new_rec()
    for i in range(T):
        r["t"].append(1000.0 - 50 * i)
        r["v_norm"].append(1.0 + 0.01 * i)
        if i == 0:
            for key in ("dt_update", "dt_pair", "k", "k_0", "r_norm", "blk_k", "blk_rdt"):
                r[key].append(50.0 if key == "dt_update" else None)
        else:
            r["dt_update"].append(50.0)
            r["dt_pair"].append(50.0)
            r["k"].append(float(global_k))
            r["k_0"].append(0.0)
            r["r_norm"].append(float(rng.uniform(0.0, 1.0)))
            r["blk_k"].append([blk_k] * B)
            r["blk_rdt"].append([blk_alpha] * B)
    return {"calib_recorder": r}


def make_image_record(seed, T=12, global_k=1e-5, global_alpha=1e-4):
    """Image record with scalar trajectory statistics."""
    r = _new_rec()
    for i in range(T):
        r["t"].append(1000.0 - 50 * i)
        r["v_norm"].append(1.0 + 0.01 * i)
        if i == 0:
            for key in ("dt_update", "dt_pair", "k", "k_0", "r_norm", "blk_k", "blk_rdt"):
                r[key].append(50.0 if key == "dt_update" else None)
        else:
            r["dt_update"].append(50.0)
            r["dt_pair"].append(50.0)
            r["k"].append(float(global_k))
            r["k_0"].append(0.0)
            r["r_norm"].append(float(global_alpha))
            r["blk_k"].append(None)
            r["blk_rdt"].append(None)
    return {"calib_recorder": r}


class Smooth(nn.Module):
    """Deterministic, near-linear denoiser -> large h_max -> real skips happen."""

    def forward(self, x, t):  # noqa: ARG002
        return x * 0.999


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_config_defaults():
    cfg = tacache.TACacheConfig(num_steps=1)
    assert cfg.cache_mode == "auto"
    assert cfg.t_blocks == 5
    assert cfg.temporal_axis == 2
    assert cfg.sample_axis == 0


def test_mode_routing():
    cfg = tacache.TACacheConfig(num_steps=1)
    assert tacache._resolve_mode([torch.randn(1, 4, 21, 8, 8)], cfg) == "video"
    assert tacache._resolve_mode([torch.randn(2, 4, 16, 16)], cfg) == "image"
    assert tacache._resolve_mode([torch.randn(2, 16, 64)], cfg) == "global"
    assert tacache._resolve_mode([torch.randn(16)], cfg) == "global"
    # explicit override wins over auto
    forced = tacache.TACacheConfig(num_steps=1, cache_mode="global")
    assert tacache._resolve_mode([torch.randn(1, 4, 21, 8, 8)], forced) == "global"


def test_video_blocks_are_frames():
    cfg = tacache.TACacheConfig(num_steps=1)  # t_blocks=5, temporal_axis=2
    x = torch.randn(1, 4, 21, 16, 16)     # T=21 frames (latent)
    regions = list(tacache._geometry_groups(x, "video", cfg))
    groups = [(index[2].start, index[2].stop) for _, index in regions]
    assert len(groups) == 5, f"expected 5 frame blocks, got {len(groups)}"
    assert groups[0][0] == 0 and groups[-1][1] == 21
    # contiguous, non-overlapping cover of all 21 frames
    assert all(groups[i][1] == groups[i + 1][0] for i in range(len(groups) - 1))
    assert groups == [(0, 4), (4, 8), (8, 12), (12, 16), (16, 21)]
    # degrade gracefully when T < t_blocks
    g2 = list(tacache._geometry_groups(torch.randn(1, 4, 3, 8, 8), "video", cfg))
    assert len(g2) == 3


def test_image_groups_are_samples():
    cfg = tacache.TACacheConfig(num_steps=1)
    regions = list(tacache._geometry_groups(torch.randn(3, 4, 16, 16), "image", cfg))
    groups = [(index[0].start, index[0].stop) for _, index in regions]
    assert len(groups) == 3
    assert groups == [(0, 1), (1, 2), (2, 3)]


def test_grouped_perp_unit_and_orthogonal():
    cfg = tacache.TACacheConfig(num_steps=1)
    ref = torch.randn(16)
    x = torch.randn(16)
    out = tacache._grouped_perp([x], [ref], "global", cfg)[0]
    assert abs(float(out.norm()) - 1.0) < 1e-4               # normalized
    assert abs(float((out * ref).sum())) < 1e-4             # orthogonal to ref
    # video: each frame block independently unit-norm + orthogonal
    v = torch.randn(1, 2, 10, 4, 4)
    rv = torch.randn(1, 2, 10, 4, 4)
    ov = tacache._grouped_perp([v], [rv], "video", cfg)[0]
    groups = [(index[2].start, index[2].stop)
              for _, index in tacache._geometry_groups(v, "video", cfg)]
    for (t0, t1) in groups:
        blk = ov[:, :, t0:t1]
        rblk = rv[:, :, t0:t1]
        assert abs(float(blk.reshape(-1).norm()) - 1.0) < 1e-4
        assert abs(float((blk.reshape(-1) * rblk.reshape(-1)).sum())) < 1e-3


def test_table_shapes_video_vs_image():
    recs_v = [make_video_record(0), make_video_record(1)]
    tab_v = tacache.build_skip_timestep_from_calib_records(
        recs_v, tau_k=0.03, tau_theta=0.30, h_cap=3)
    blk = np.array(tab_v["k_off_used"])
    assert blk.ndim == 2 and blk.shape[1] == 5
    assert len(tab_v["h_max_off"]) == blk.shape[0]
    assert np.array(tab_v["r_dt_over_v_off_used"]).shape == blk.shape

    recs_i = [make_image_record(0), make_image_record(1)]
    tab_i = tacache.build_skip_timestep_from_calib_records(
        recs_i, tau_k=0.03, tau_theta=0.30, h_cap=3)
    assert np.array(tab_i["k_off_used"]).ndim == 1
    assert np.array(tab_i["r_dt_over_v_off_used"]).ndim == 1


def test_normalise_table_rank():
    recs_v = [make_video_record(0), make_video_record(1)]
    recs_i = [make_image_record(0), make_image_record(1)]
    tmp = tempfile.mkdtemp()
    try:
        pv = os.path.join(tmp, "v.json")
        pi = os.path.join(tmp, "i.json")
        json.dump(tacache.build_skip_timestep_from_calib_records(
            recs_v, tau_k=0.03, tau_theta=0.30, h_cap=3), open(pv, "w"))
        json.dump(tacache.build_skip_timestep_from_calib_records(
            recs_i, tau_k=0.03, tau_theta=0.30, h_cap=3), open(pi, "w"))
        assert tacache.load_skip_table(pv)[0]["k_off_used"].ndim == 2   # video -> per-block
        assert tacache.load_skip_table(pi)[0]["k_off_used"].ndim == 1   # image -> scalar
        malformed = json.load(open(pv))
        malformed["k_off_used"] = np.mean(malformed["k_off_used"], axis=1).tolist()
        malformed["r_dt_over_v_off_used"] = np.mean(malformed["r_dt_over_v_off_used"], axis=1).tolist()
        json.dump(malformed, open(pv, "w"))
        with np.testing.assert_raises(ValueError):
            tacache.load_skip_table(pv)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_transition_index_alignment():
    """table[i] represents the future transition i -> i+1."""
    record = _new_rec()
    record["t"] = [3.0, 2.0, 1.0, 0.0]
    record["dt_update"] = [1.0, 1.0, 1.0, None]
    record["dt_pair"] = [None, 1.0, 1.0, 1.0]
    record["v_norm"] = [1.0, 1.0, 1.0, 1.0]
    # record[j] is measured only after v_j exists, hence it describes j-1 -> j.
    record["k"] = [None, 10.0, 20.0, 30.0]
    record["k_0"] = [None, 0.0, 0.0, 0.0]
    record["r_norm"] = [None, 0.1, 0.2, 0.3]
    record["blk_k"] = [None, None, None, None]
    record["blk_rdt"] = [None, None, None, None]
    table = tacache.build_skip_timestep_from_calib_records(
        [{"calib_recorder": record}],
        tau_k=100.0, tau_theta=100.0, h_cap=4, lambda_std=0.0,
    )
    # The builder shifts record statistics left by one: table[0] = transition 0 -> 1.
    assert np.allclose(table["k_off_used"], [10.0, 20.0, 30.0])
    assert np.allclose(table["r_dt_over_v_off_used"], [0.1, 0.2, 0.3])
    assert np.allclose(table["dt_update_used"], [1.0, 1.0, 1.0])


def test_h_source_video_uses_block_mean():
    # global k tiny (would allow long skips) but per-block k LARGE.
    # If h came from global k -> h_cap; if from block mean -> 1.
    recs = [make_video_record(s, global_k=1e-6, blk_k=0.05, blk_alpha=0.5)
            for s in range(2)]
    tab = tacache.build_skip_timestep_from_calib_records(
        recs, tau_k=0.03, tau_theta=0.30, h_cap=3)
    assert max(tab["h_max_off"]) == 1, "video h must be driven by the per-block MEAN"

    # opposite: global k LARGE but per-block tiny -> block-mean drives h high.
    recs2 = [make_video_record(s, global_k=0.05, blk_k=1e-6, blk_alpha=1e-5)
             for s in range(2)]
    tab2 = tacache.build_skip_timestep_from_calib_records(
        recs2, tau_k=0.03, tau_theta=0.30, h_cap=3)
    assert max(tab2["h_max_off"]) > 1, "video h must ignore the global scalar"


def test_h_source_image_uses_global_scalar():
    recs = [make_image_record(s, global_k=1e-6, global_alpha=1e-5) for s in range(2)]
    tab = tacache.build_skip_timestep_from_calib_records(
        recs, tau_k=0.03, tau_theta=0.30, h_cap=3)
    assert max(tab["h_max_off"]) > 1, "image h should be driven by the (tiny) global k"

    recs2 = [make_image_record(s, global_k=0.05, global_alpha=0.5) for s in range(2)]
    tab2 = tacache.build_skip_timestep_from_calib_records(
        recs2, tau_k=0.03, tau_theta=0.30, h_cap=3)
    assert max(tab2["h_max_off"]) == 1, "large global k must shrink image h to 1"


def test_roundtrip_video():
    m = Smooth()
    tmp = tempfile.mkdtemp()

    def gen():
        x = torch.randn(1, 4, 21, 16, 16)
        for s in range(20):
            x = m.forward(x, torch.tensor(float(1000 - 50 * s)))
        return x

    try:
        cal = calibrate(
            m, gen, config=tacache.TACacheConfig(
                num_steps=20, output_prefix=os.path.join(tmp, "vid"), verbose=False))
        st = json.load(open(cal.skip_path))
        assert np.array(st["k_off_used"]).shape[1] == 5
        gn = infer(
            m, gen, skip_path=cal.skip_path,
            config=tacache.TACacheConfig(num_steps=20, verbose=False))
        assert gn.skip_count > 0, "video inference produced no skips"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_roundtrip_image():
    m = Smooth()
    tmp = tempfile.mkdtemp()

    def gen():
        x = torch.randn(3, 4, 16, 16)
        for s in range(20):
            x = m.forward(x, torch.tensor(float(1000 - 50 * s)))
        return x

    try:
        cal = calibrate(
            m, gen, config=tacache.TACacheConfig(
                num_steps=20, output_prefix=os.path.join(tmp, "img"), verbose=False))
        st = json.load(open(cal.skip_path))
        assert np.array(st["k_off_used"]).ndim == 1
        gn = infer(
            m, gen, skip_path=cal.skip_path,
            config=tacache.TACacheConfig(num_steps=20, verbose=False))
        assert gn.skip_count > 0, "image inference produced no skips"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_inference_preserves_shape():
    m = Smooth()
    tmp = tempfile.mkdtemp()

    def gen():
        x = torch.randn(1, 4, 21, 8, 8)
        for s in range(20):
            x = m.forward(x, torch.tensor(float(1000 - 50 * s)))
        return x

    try:
        cal = calibrate(
            m, gen, config=tacache.TACacheConfig(
                num_steps=20, output_prefix=os.path.join(tmp, "shp"), verbose=False))
        res = infer(
            m, gen, skip_path=cal.skip_path,
            config=tacache.TACacheConfig(num_steps=20, verbose=False))
        assert tuple(res.output.shape) == (1, 4, 21, 8, 8)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cfg_combined_caching():
    # External CFG combines two forwards per step. Cache the combined velocity
    # and skip both branches together.
    G = 5.0

    class CFGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x, t=None, ctx=0.0):
            self.calls += 1
            return [x[0] * 0.9999 + 0.0005 * ctx]

    m = CFGNet()
    tmp = tempfile.mkdtemp()

    def gen():
        x = [torch.randn(4, 21, 8, 8)]
        for i in range(20):
            t = torch.tensor([float(1000 - 50 * i)])
            c = m.forward(x, t=t, ctx=1.0)[0]
            u = m.forward(x, t=t, ctx=0.0)[0]
            x = [x[0] - (u + G * (c - u)) * 0.05]
        return x[0]

    cfg = tacache.TACacheConfig(
        num_steps=40, num_streams=2, cfg_scale=G,
        cache_mode="video", temporal_axis=2,
        tau_k=10.0, tau_theta=10.0, h_cap=6,
        output_prefix=os.path.join(tmp, "cfg"), verbose=False)
    try:
        cal = calibrate(m, gen, config=cfg)
        m.calls = 0
        gn = infer(m, gen, skip_path=cal.skip_path, config=cfg)
        assert gn.skip_count > 0, "no skips happened (thresholds too tight)"
        assert m.calls == 2 * (20 - gn.skip_count), (m.calls, gn.skip_count)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_auto_sigma_video():
    # With native times as the available sequence, each geometry uses native t.
    for mode in ("video", "image", "global"):
        wrapper = tacache.TACacheWrapper(Smooth(), tacache.TACacheConfig(num_steps=2, cache_mode=mode))
        assert wrapper._effective_sigma_fn() is None


def main():
    tests = [
        ("config defaults", test_config_defaults),
        ("mode routing (auto + override)", test_mode_routing),
        ("video blocks are frame segments", test_video_blocks_are_frames),
        ("image groups are samples", test_image_groups_are_samples),
        ("grouped perp: unit-norm + orthogonal", test_grouped_perp_unit_and_orthogonal),
        ("table shapes: video 2D vs image 1D", test_table_shapes_video_vs_image),
        ("normalise_table rank (2D/1D)", test_normalise_table_rank),
        ("transition index alignment", test_transition_index_alignment),
        ("h source: video uses block mean", test_h_source_video_uses_block_mean),
        ("h source: image uses global scalar", test_h_source_image_uses_global_scalar),
        ("round trip: video (per-block skip)", test_roundtrip_video),
        ("round trip: image (per-sample skip)", test_roundtrip_image),
        ("inference preserves output shape", test_inference_preserves_shape),
        ("CFG combined caching (skip avoids both branches)", test_cfg_combined_caching),
        ("geometry does not choose time units", test_auto_sigma_video),
    ]
    for name, fn in tests:
        check(name, fn)

    passed = sum(_RESULTS)
    total = len(_RESULTS)
    print(f"\n==== {passed}/{total} passed ====")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
