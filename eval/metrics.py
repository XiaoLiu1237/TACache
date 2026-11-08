"""Frame-aligned PSNR, SSIM and LPIPS for original-model and TACache videos."""
import argparse
import json
from pathlib import Path
import re

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F


@torch.inference_mode()
def evaluate_video(reference_path, generated_path, model, window, device, batch_size):
    reference_capture = cv2.VideoCapture(str(reference_path))
    generated_capture = cv2.VideoCapture(str(generated_path))
    total = np.zeros(3, dtype=np.float64)
    frames = 0
    try:
        if not reference_capture.isOpened() or not generated_capture.isOpened():
            raise ValueError(f"Cannot open video pair: {reference_path}, {generated_path}")
        while True:
            references, generated = [], []
            for _ in range(batch_size):
                ref_ok, ref = reference_capture.read()
                gen_ok, gen = generated_capture.read()
                if ref_ok != gen_ok:
                    raise ValueError(f"Frame counts differ: {reference_path}, {generated_path}")
                if not ref_ok:
                    break
                if ref.shape != gen.shape:
                    raise ValueError(f"Frame sizes differ: {reference_path}, {generated_path}")
                references.append(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
                generated.append(cv2.cvtColor(gen, cv2.COLOR_BGR2RGB))
            if not references:
                break

            x = torch.from_numpy(np.stack(references)).to(device=device, dtype=torch.float32)
            y = torch.from_numpy(np.stack(generated)).to(device=device, dtype=torch.float32)
            x = x.permute(0, 3, 1, 2).div_(255.0)
            y = y.permute(0, 3, 1, 2).div_(255.0)

            # RGB PSNR and SSIM with an 11x11 Gaussian window (sigma=1.5).
            mse = (x - y).square().mean(dim=(1, 2, 3))
            psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
            mu_x = F.conv2d(x, window, groups=3)
            mu_y = F.conv2d(y, window, groups=3)
            mu_x_sq, mu_y_sq, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
            sigma_x = F.conv2d(x.square(), window, groups=3) - mu_x_sq
            sigma_y = F.conv2d(y.square(), window, groups=3) - mu_y_sq
            sigma_xy = F.conv2d(x * y, window, groups=3) - mu_xy
            ssim = ((2 * mu_xy + 0.01**2) * (2 * sigma_xy + 0.03**2)) / (
                (mu_x_sq + mu_y_sq + 0.01**2) * (sigma_x + sigma_y + 0.03**2)
            )
            ssim = ssim.mean(dim=(1, 2, 3))
            perceptual = model(x * 2 - 1, y * 2 - 1).flatten()
            total += [psnr.sum().item(), ssim.sum().item(), perceptual.sum().item()]
            frames += len(references)
    finally:
        reference_capture.release()
        generated_capture.release()
    if frames == 0:
        raise ValueError(f"No frames decoded: {generated_path}")
    return {
        "frames": frames,
        "psnr": float(total[0] / frames),
        "ssim": float(total[1] / frames),
        "lpips": float(total[2] / frames),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt_video_dir", type=Path, required=True)
    parser.add_argument("--generated_video_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8, help="Frames per metric batch")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    videos = sorted(args.generated_video_dir.rglob("*.mp4"))
    if not videos:
        parser.error(f"No MP4 videos found in {args.generated_video_dir}")

    # Match relative paths; also accept run.py's baseline_ / infer_ filenames.
    pairs = []
    for video in videos:
        relative = video.relative_to(args.generated_video_dir)
        reference = args.gt_video_dir / relative
        if not reference.is_file() and re.fullmatch(r"infer_[0-9]+\.mp4", relative.name):
            reference = reference.with_name(relative.name.replace("infer_", "baseline_", 1))
        if not reference.is_file():
            raise FileNotFoundError(f"Missing reference for {relative}: {reference}")
        pairs.append((reference, video, relative.as_posix()))

    torch.set_num_threads(1)
    device = torch.device(args.device)
    model = lpips.LPIPS(net="alex").to(device).eval()
    coords = torch.arange(11, device=device, dtype=torch.float32) - 5
    kernel = torch.exp(-(coords**2) / (2 * 1.5**2))
    kernel /= kernel.sum()
    window = torch.outer(kernel, kernel).expand(3, 1, 11, 11).contiguous()

    results = []
    for reference, video, name in pairs:
        result = evaluate_video(reference, video, model, window, device, args.batch_size)
        results.append(dict(video=name, **result))
        print(f"[{len(results)}/{len(pairs)}] {name}: "
              f"PSNR={result['psnr']:.6f} SSIM={result['ssim']:.6f} "
              f"LPIPS={result['lpips']:.6f}", flush=True)
    summary = {key: sum(row[key] for row in results) / len(results)
               for key in ("psnr", "ssim", "lpips")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "videos": results}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
