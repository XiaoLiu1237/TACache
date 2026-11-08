# Reproduction

## Installation

Requires Python 3.10 and an NVIDIA CUDA GPU.

```bash
conda create -n tacache python=3.10 -y
conda activate tacache
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn==2.5.8 --no-build-isolation
```

FlashAttention requires a CUDA toolkit. Clone the Wan source:

```bash
git clone https://github.com/Wan-Video/Wan2.1.git Wan2.1/Wan2.1
git -C Wan2.1/Wan2.1 checkout ae487cc653b4a1791fec8201af20d2102a2514f3
```

For an existing checkout, use `--wan-root`. Download the [checkpoint](https://github.com/Wan-Video/Wan2.1) into `models/Wan2.1-T2V-1.3B/`.

## Quickstart

```bash
bash scripts/quickstart.sh "A small boat sailing on a calm lake."
```

The script calibrates on the supplied prompt, builds a skip table, and generates one TACache video in `outputs/quickstart/inference/`. The calibration files are saved in `outputs/quickstart/calibration/`.

Additional Wan CLI options, such as `--checkpoint`, `--wan-root`, or `--seed`, apply to both stages. For full runs, use a representative calibration prompt set as described below.

## Data Preparation

### Calibration data

For Wan2.1, we use **30 text prompts from [InternVid-10M-FLT](https://huggingface.co/datasets/OpenGVLab/InternVid)**, selected by aesthetic quality and prompt category as described in Appendix A. The exact prompts used in our experiments are provided in [Wan2.1/inputs/calibration_prompts.txt](Wan2.1/inputs/calibration_prompts.txt), one prompt per line.

### Evaluation data

Download the [VBench evaluation prompts](https://github.com/Vchitect/VBench/tree/master/prompts/prompts_per_dimension) and place the `.txt` files in `data/vbench/`.

## Calibration and Inference

The following commands use [Wan2.1/configs/default.json](Wan2.1/configs/default.json) for the Wan2.1-1.3B setting in **Table 2 (TACache)**. Run from the repository root.

### Calibration

```bash
# Calibrate on 30 InternVid prompts for Table 2 (Wan2.1-1.3B, TACache)
python Wan2.1/run.py --mode calibrate \
  --prompts-file Wan2.1/inputs/calibration_prompts.txt \
  --outdir outputs/wan2_1/calibration
```

After all 30 prompts finish, use `outputs/wan2_1/calibration/tacache_skip_timestep.json` for inference. Calibration records and generated calibration videos are saved alongside it.

### Evaluation video generation

```bash
# Generate evaluation videos for Table 2 (Wan2.1-1.3B, TACache)
for split in overall_consistency scene subject_consistency; do
  python Wan2.1/run.py --mode infer \
    --prompts-file "data/vbench/${split}.txt" \
    --skip-table outputs/wan2_1/calibration/tacache_skip_timestep.json \
      --outdir "outputs/wan2_1/vbench/${split}"
done
```

### Your own prompts

For a single video, reuse the completed table:

```bash
python Wan2.1/run.py --mode infer \
  --prompt "A small boat on a calm lake at sunrise, cinematic." \
  --skip-table outputs/wan2_1/calibration/tacache_skip_timestep.json \
  --outdir outputs/wan2_1/inference
```

For a different calibration set, supply a UTF-8 text file with one prompt per line through `--prompts-file`. Keep the checkpoint, sampling settings, seed, and TACache configuration aligned between calibration and inference. To change the configuration, copy `Wan2.1/configs/default.json`, pass `--config your_config.json` to both stages, and recalibrate.

## Evaluation

Install the optional image-quality metrics with:

```bash
python -m pip install -r requirements-eval.txt
```

[VBench](https://github.com/Vchitect/VBench#usage) and [Q-Align](https://github.com/Q-Future/Q-Align) use separate environments. The Wan evaluation protocol is:

| Metric | Inputs |
| --- | --- |
| VBench | VBench 0.1.5 standard mode. |
| PSNR / SSIM / LPIPS (AlexNet) | All 81 decoded frames, paired by prompt and frame index with matching reference videos. |
| CLIP-IQA / Q-Align | Frames 0, 5, …, 80: 17 frames per video. |

For fidelity and image-quality scores, average over frames within each video, then equally over videos. These commands reproduce the calibration and video-generation stage; scoring requires the corresponding metric setup and, for fidelity metrics, reference videos.
