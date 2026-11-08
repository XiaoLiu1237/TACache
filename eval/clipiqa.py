"""Evaluate CLIP-IQA's Good-photo probability on every fifth video frame."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import cv2
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_dir", type=Path, required=True)
    parser.add_argument("--clipiqa_root", type=Path, default=Path("third_party/CLIP-IQA"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("outputs/clipiqa.json"))
    args = parser.parse_args()
    videos = sorted(args.video_dir.rglob("*.mp4"))
    if not videos:
        parser.error(f"No MP4 videos found in {args.video_dir}")

    sys.path.insert(0, str(args.clipiqa_root.resolve()))
    from mmedit.apis import init_model, restoration_inference

    torch.set_num_threads(1)
    config = args.clipiqa_root / "configs/clipiqa/clipiqa_attribute_test.py"
    model = init_model(str(config), None, device=torch.device(args.device))
    results = []
    with tempfile.TemporaryDirectory() as temporary:
        frame_path = str(Path(temporary) / "frame.png")
        for video in videos:
            capture = cv2.VideoCapture(str(video))
            scores, index = [], 0
            try:
                if not capture.isOpened():
                    raise ValueError(f"Cannot open video: {video}")
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if index % 5 == 0:
                        if not cv2.imwrite(frame_path, frame):
                            raise OSError(f"Cannot save sampled frame: {frame_path}")
                        _, attributes = restoration_inference(model, frame_path, return_attributes=True)
                        scores.append(float(attributes.float().detach().cpu().numpy()[0][0]))
                    index += 1
            finally:
                capture.release()
            if not scores:
                raise ValueError(f"No frames decoded: {video}")
            score = sum(scores) / len(scores)
            results.append(dict(video=video.relative_to(args.video_dir).as_posix(),
                                frames=len(scores), clipiqa=score))
            print(f"[{len(results)}/{len(videos)}] {video.name}: CLIP-IQA={score:.8f}", flush=True)

    summary = {"clipiqa": sum(row["clipiqa"] for row in results) / len(results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "videos": results}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
