"""Generate Wan videos or run a calibration prompt set with TACache."""
import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tacache import read_prompts_file
from tacache import TACacheConfig, save_calibration
from tacache.utils import write_json
from tacache.utils import add_arguments, cache_options, parse_configured_args
from tacache.utils import normalize_path_arguments, project_working_directory

@project_working_directory()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--task", choices=("t2v-1.3B", "t2v-14B"), default="t2v-1.3B")
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--shift", type=float, default=5.)
    parser.add_argument("--solver", choices=("unipc", "dpm++"), default="unipc")
    parser.add_argument("--guidance", type=float, default=5.)
    parser.add_argument("--wan-root", default="Wan2.1/Wan2.1", help="Path to the official Wan2.1 checkout")
    parser.set_defaults(mode="infer")
    for action in parser._actions:
        if action.dest == "outdir":
            action.required = False
    args = normalize_path_arguments(parse_configured_args(parser, argv, "Wan2.1/configs/default.json"))
    if args.outdir is None:
        args.outdir = Path("outputs/wan2_1") / args.mode
    if args.mode == "calibrate":
        args.skip_table = None
    options = cache_options(args)
    if not args.checkpoint:
        parser.error("--checkpoint is required")
    if args.mode == "infer" and not args.skip_table:
        parser.error("infer requires --skip-table")

    # Load Wan and prepare the prompts.
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(Path(args.wan_root).resolve()))
    try:
        import wan
    except ModuleNotFoundError as error:
        if error.name == "wan":
            raise RuntimeError("Wan2.1 source not found; clone Wan2.1 or pass --wan-root") from error
        raise
    from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
    import tacache_wan
    pipeline = wan.WanT2V(config=WAN_CONFIGS[args.task], checkpoint_dir=args.checkpoint,
        device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=False)
    prompts = [args.prompt] if args.prompt is not None else read_prompts_file(args.prompts_file)
    if not prompts:
        raise ValueError("Prompt set is empty")
    args.outdir.mkdir(parents=True, exist_ok=True)
    calibrations, runs = [], []
    config = TACacheConfig(num_steps=2*args.steps, num_streams=2, cfg_scale=args.guidance,
        model_family="wan2_1", workflow="t2v", require_table_metadata=True,
        time_coordinate="sigma", timestep_semantics="native_t",
        sigma_fn_id=f"wan_{args.solver}_indexed_sigmas_v1",
        calibration_context={"workflow":"t2v", "size":args.size, "frames":args.frames,
            "shift":args.shift, "solver":args.solver, "task":args.task,
            "dtype":"bfloat16", "checkpoint":args.checkpoint.rstrip('/').split('/')[-1],
            "seed":args.seed, "segment_history":"next_prediction"}, **options)
    with torch.inference_mode():
        for index, prompt in enumerate(prompts):
            local = replace(config, output_prefix=str(args.outdir / "calibration_records" / f"{index:04d}"))
            video, cache = tacache_wan.generate(pipeline, prompt, mode=args.mode,
                config=local, skip_table=args.skip_table, size=SIZE_CONFIGS[args.size],
                frame_num=args.frames, shift=args.shift, sample_solver=args.solver,
                sampling_steps=args.steps, guide_scale=args.guidance, seed=args.seed,
                offload_model=False)
            # Save the generated video and, for calibration, update the table.
            path = args.outdir / f"{args.mode}_{index:04d}.mp4"
            tacache_wan.save_video(video, path)
            runs.append(dict(prompt_index=index, artifact=str(path), **cache.report))
            if args.mode == "calibrate":
                calibrations.append(cache.calibration)
                save_calibration(calibrations, args.outdir, config)
            write_json(args.outdir / "run_summary.json", dict(model="wan2_1", mode=args.mode,
                settings=dict(task=args.task, checkpoint=args.checkpoint, seed=args.seed,
                    size=args.size, frames=args.frames, steps=args.steps, guidance=args.guidance,
                    shift=args.shift, solver=args.solver, offload_model=False, fps=16, cpu_threads=1,
                    video_uint8_conversion="truncate",
                    tau_k=args.tau_k, tau_theta=args.tau_theta, h_cap=args.h_cap,
                    lambda_std=args.lambda_std, t_blocks=args.t_blocks,
                    time_coordinate="scheduler_sigma", segment_history="next_prediction",
                    skip_table=args.skip_table), runs=runs))


if __name__ == "__main__":
    main()
