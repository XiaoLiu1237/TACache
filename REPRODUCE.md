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

## Calibration and Inference

Run the commands below from the repository root. Calibration builds a skip table once; inference reuses that table for new prompts.

### 1. Prepare calibration prompts

Create a UTF-8 file named `my_calibration_prompts.txt`, with one prompt per line:

```text
A red fox walking through a snowy forest.
A small boat sailing on a calm lake.
```

These lines illustrate the file format. For your experiment, choose prompts that cover the subjects, scenes, and motion you want to generate. Calibration needs text prompts; reference images or videos are not required. Blank lines and lines starting with `#` are ignored.

The quickstart uses one prompt to check the workflow. Use your representative calibration set for the following steps.

### 2. Run calibration and build the table

```bash
CUDA_VISIBLE_DEVICES=0 python Wan2.1/run.py --mode calibrate \
  --prompts-file my_calibration_prompts.txt \
  --seed 0 \
  --outdir outputs/wan2_1/calibration
```

This runs the model on every calibration prompt, records the velocity statistics, and automatically builds the skip table. You do not need to create or edit the table JSON yourself.

The calibration directory contains:

| File | Contents |
| --- | --- |
| `tacache_skip_timestep.json` | The skip table used by TACache inference. |
| `tacache_calib_records.json` | Aggregated calibration records. |
| `run_summary.json` | Generation settings and run statistics. |
| `calibrate_0000.mp4`, ... | Videos generated during calibration, in prompt order. |

Wait for calibration to finish before using its table; the table is updated as prompts are processed.

### 3. Generate a video with TACache

Pass the generated table through `--skip-table`:

```bash
CUDA_VISIBLE_DEVICES=0 python Wan2.1/run.py --mode infer \
  --prompt "A small boat on a calm lake at sunrise, cinematic." \
  --skip-table outputs/wan2_1/calibration/tacache_skip_timestep.json \
  --seed 0 \
  --outdir outputs/wan2_1/inference
```

The video is saved as `outputs/wan2_1/inference/infer_0000.mp4`, with its settings and run statistics in `run_summary.json`. The inference prompt can differ from the calibration prompts.

To generate multiple videos, create `my_inference_prompts.txt` in the same one-prompt-per-line format and use `--prompts-file` instead of `--prompt`:

```bash
CUDA_VISIBLE_DEVICES=0 python Wan2.1/run.py --mode infer \
  --prompts-file my_inference_prompts.txt \
  --skip-table outputs/wan2_1/calibration/tacache_skip_timestep.json \
  --seed 0 \
  --outdir outputs/wan2_1/inference_batch
```

Prompts are processed in file order, producing `infer_0000.mp4`, `infer_0001.mp4`, and so on. Each prompt uses the supplied seed independently.

### 4. Keep calibration and inference settings aligned

Both stages read [Wan2.1/configs/default.json](Wan2.1/configs/default.json). Command-line options override the JSON settings.

For a custom setup, copy that file to `my_wan_config.json`, edit it, and pass `--config my_wan_config.json` to both commands. Set checkpoint and Wan source locations with `--checkpoint` and `--wan-root` when needed.

Use the same checkpoint, sampling settings, seed, and TACache configuration when building and using a table. If you change these settings, rerun calibration to generate a matching table. You can reuse the completed table across different inference prompts with the same configuration.

## Evaluation

Optional metrics:

```bash
python -m pip install -r requirements-eval.txt
```

Includes LPIPS, PyIQA and CLIP-IQA. [VBench](https://github.com/Vchitect/VBench) and [Q-Align](https://github.com/Q-Future/Q-Align) require separate environments.
