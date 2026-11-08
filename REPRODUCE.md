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

## Calibration and Inference

Create `my_calibration_prompts.txt` with one representative prompt per line:

```text
A red fox walking through a snowy forest.
A small boat sailing on a calm lake.
```

Run from the project directory:

```bash
CUDA_VISIBLE_DEVICES=0 python Wan2.1/run.py --mode calibrate \
  --prompts-file my_calibration_prompts.txt \
  --outdir outputs/wan2_1/calibration

CUDA_VISIBLE_DEVICES=0 python Wan2.1/run.py --mode infer \
  --prompt 'A small boat on a calm lake at sunrise, cinematic.' \
  --skip-table outputs/wan2_1/calibration/tacache_skip_timestep.json
```

Prepare your own calibration data and skip table. Keep model, sampling settings and seed consistent between calibration and inference. Use `--mode baseline` for original inference.

## Evaluation

Optional metrics:

```bash
python -m pip install -r requirements-eval.txt
```

Includes LPIPS, PyIQA and CLIP-IQA. [VBench](https://github.com/Vchitect/VBench) and [Q-Align](https://github.com/Q-Future/Q-Align) require separate environments.
