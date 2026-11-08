# Reproduction

## Installation

Requires Python 3.10, an NVIDIA GPU, and a CUDA toolkit. Run from the repository root.

```bash
conda create -n tacache python=3.10 -y
conda activate tacache
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn==2.5.8 --no-build-isolation

git clone https://github.com/Wan-Video/Wan2.1.git Wan2.1/Wan2.1
git -C Wan2.1/Wan2.1 checkout ae487cc653b4a1791fec8201af20d2102a2514f3
```

Download the [Wan2.1 checkpoint](https://github.com/Wan-Video/Wan2.1) into `models/Wan2.1-T2V-1.3B/`.

## Data Preparation

- **Calibration:** [InternVid-10M-FLT](https://huggingface.co/datasets/OpenGVLab/InternVid). The calibration prompts are provided in [Wan2.1/inputs/calibration_prompts.txt](Wan2.1/inputs/calibration_prompts.txt).
- **Evaluation:** Download the [VBench prompt lists](https://github.com/Vchitect/VBench/tree/master/prompts/prompts_per_dimension) and place the `.txt` files in `data/vbench/`.

## Calibration and Inference

Wan2.1-1.3B, TACache in Table 2:

```bash
# Calibration
python Wan2.1/run.py --mode calibrate \
  --prompts-file Wan2.1/inputs/calibration_prompts.txt \
  --outdir outputs/wan2_1/calibration

# Generate evaluation videos
for split in overall_consistency scene subject_consistency; do
  python Wan2.1/run.py --mode infer \
    --prompts-file "data/vbench/${split}.txt" \
    --skip-table outputs/wan2_1/calibration/tacache_skip_timestep.json \
    --outdir "outputs/wan2_1/vbench/${split}"
done
```

For a custom prompt:

```bash
python Wan2.1/run.py --mode infer \
  --prompt "A small boat on a calm lake at sunrise, cinematic." \
  --skip-table outputs/wan2_1/calibration/tacache_skip_timestep.json \
  --outdir outputs/wan2_1/inference
```
