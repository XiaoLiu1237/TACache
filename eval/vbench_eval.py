"""Evaluate the eight VBench dimensions used for Wan in Table 2."""
import argparse
import json
from pathlib import Path

import torch


GROUPS = {
    "overall_consistency": ["overall_consistency", "aesthetic_quality", "imaging_quality"],
    "scene": ["scene", "background_consistency"],
    "subject_consistency": ["subject_consistency", "dynamic_degree", "motion_smoothness"],
}
RANGES = {
    "subject_consistency": (0.1462, 1.0),
    "background_consistency": (0.2615, 1.0),
    "motion_smoothness": (0.706, 0.9975),
    "dynamic_degree": (0.0, 1.0),
    "aesthetic_quality": (0.0, 1.0),
    "imaging_quality": (0.0, 1.0),
    "scene": (0.0, 0.8222),
    "overall_consistency": (0.0, 0.364),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_dir", type=Path, required=True)
    parser.add_argument("--prompts_dir", type=Path, default=Path("data/vbench"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/vbench"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import vbench
    from vbench import VBench

    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((Path(vbench.__file__).parent / "VBench_full_info.json").read_text())
    counts = {}
    for group in GROUPS:
        prompts = [line.strip() for line in (args.prompts_dir / f"{group}.txt").read_text().splitlines()
                   if line.strip() and not line.lstrip().startswith("#")]
        if not prompts:
            raise ValueError(f"Empty prompt list: {group}")
        available = {row["prompt_en"]: row for row in metadata if group in row["dimension"]}
        selected = [available[prompt] for prompt in prompts]
        (args.output_dir / f"{group}_prompts.json").write_text(json.dumps(selected, indent=2))
        counts[group] = len(prompts)
        folder = args.output_dir / "input_links" / group
        folder.mkdir(parents=True, exist_ok=True)
        for index, prompt in enumerate(prompts):
            video = args.video_dir / group / f"infer_{index:04d}.mp4"
            if not video.is_file():
                video = args.video_dir / group / f"{prompt}-0.mp4"
            if not video.is_file():
                raise FileNotFoundError(f"Missing video for {group}, prompt {index}: {prompt}")
            # The experiment uses the same seed for all five VBench repetitions.
            for repeat in range(5):
                link = folder / f"{prompt}-{repeat}.mp4"
                if link.is_symlink():
                    link.unlink()
                link.symlink_to(video.resolve())

    raw = {}
    for group, dimensions in GROUPS.items():
        evaluator = VBench(torch.device(args.device), str(args.output_dir / f"{group}_prompts.json"),
                          str(args.output_dir))
        for dimension in dimensions:
            evaluator.evaluate(str(args.output_dir / "input_links" / group), dimension,
                               dimension_list=[dimension], mode="vbench_standard")
            result = json.loads((args.output_dir / f"{dimension}_eval_results.json").read_text())
            raw[dimension] = float(result[dimension][0])

    normalized = {name: max(0.0, min(1.0, (raw[name] - low) / (high - low)))
                  for name, (low, high) in RANGES.items()}
    weights = {name: 0.5 if name == "dynamic_degree" else 1.0 for name in RANGES}
    score = 100 * sum(normalized[name] * weights[name] for name in RANGES) / sum(weights.values())
    summary = dict(vbench_8dim_percent=score, raw=raw, normalized=normalized, prompt_counts=counts)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"VBench (eight dimensions): {score:.6f}%")
    print(f"Saved to {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
