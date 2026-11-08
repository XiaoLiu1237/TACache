"""Evaluate Q-Align image quality on every fifth video frame."""
import argparse
import json
from pathlib import Path
import sys

import cv2
from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_dir", type=Path, required=True)
    parser.add_argument("--pretrained", default="q-future/one-align")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("outputs/qalign.json"))
    args = parser.parse_args()
    videos = sorted(args.video_dir.rglob("*.mp4"))
    if not videos:
        parser.error(f"No MP4 videos found in {args.video_dir}")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/Q-Align"))
    from q_align import QAlignScorer, MPLUGOwl2LlamaForCausalLM
    from q_align.model.modeling_mplug_owl2 import MPLUGOwl2LlamaModel

    # Keep the eager attention path used by the original Q-Align evaluation.
    MPLUGOwl2LlamaForCausalLM._supports_sdpa = False
    MPLUGOwl2LlamaModel._supports_sdpa = False

    torch.set_num_threads(1)
    scorer = QAlignScorer(pretrained=args.pretrained, device=args.device)
    results = []
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        frames, index = [], 0
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open video: {video}")
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % 5 == 0:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                index += 1
        finally:
            capture.release()
        if not frames:
            raise ValueError(f"No frames decoded: {video}")
        scores = []
        for offset in range(0, len(frames), 4):
            scores.extend(float(value) for value in scorer(frames[offset:offset + 4]).float().cpu().tolist())
        score = sum(scores) / len(scores)
        results.append(dict(video=video.relative_to(args.video_dir).as_posix(),
                            frames=len(scores), qalign=score))
        print(f"[{len(results)}/{len(videos)}] {video.name}: Q-Align={score:.8f}", flush=True)

    summary = {"qalign": sum(row["qalign"] for row in results) / len(results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "videos": results}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
