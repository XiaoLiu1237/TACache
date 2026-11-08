<h1 align="center">Accelerating Rectified Flow Models  via Trajectory-Aware Caching</h1>

<p align="center">
  <a href="https://github.com/XiaoLiu1237">Xiao Liu</a>,
  <a href="https://kai-liu.cn/">Kai Liu</a>,
  <a href="https://scholar.google.com/citations?user=TVyk3f0AAAAJ&hl=en">Naiyang Guan</a>,
  <a href="https://lukebird17.github.io/">Hongliang Lu</a>,
  <a href="https://scholar.google.com/citations?user=tZS6bPMAAAAJ&hl=en&oi=sra">Zhixin Wang</a>,
  <a>Zhikai Chen</a>,
  <a href="https://scholar.google.com/citations?hl=en&user=zEEMPUUAAAAJ&view_op=list_works&sortby=pubdate">Renjing Pei</a>,
  <a href="https://yulunzhang.com/">Yulun Zhang<sup>*</sup></a>
</p>

<p align="center">
  <I>NeurIPS 2026</i>
</p>

<p align="center">
  <img alt="downloads" src="https://img.shields.io/github/downloads/XiaoLiu1237/TACache/total">
  <img alt="visitors"  src="https://visitor-badge.laobi.icu/badge?page_id=XiaoLiu1237.TACache">
  <img alt="Stars" src="https://img.shields.io/github/stars/XiaoLiu1237/TACache?style=social">
</p>

<p align="center">
  <a href="YOUR_ARXIV_URL">[arXiv]</a>
  <a href="YOUR_SUPP_URL">[supplementary material]</a>
</p>

---

## 🔥 News
- **2025-9-24:** TACache is accepted by NeurIPS 26! 🎉🎉🎉
- **2025-2-6:** Repo released. ⭐️⭐️⭐️

---

## ✨ Pipeline
<p align="center">
  <img src="fig/overview.png" width="92%">
</p>


---

## 🚀 TODO
- [ ] Release code.

---

## Contents

1. [Results](#results)
2. [Installation](#installation)
3. [Evaluation](#evaluation)
4. [Calibration and Inference](#calibration-and-inference)
5. [Citation](#citation)

---

## Results

**Wan2.1 · Text to video**

Original-model and TACache videos use the same prompt, seed and sampling settings.

| Scene | Original model | TACache |
| :--- | :---: | :---: |
| Snow fox | [![Snow fox - original](assets/showcase/wan_fox_original.gif)](assets/showcase/wan_fox_original.mp4) | [![Snow fox - tacache](assets/showcase/wan_fox_tacache.gif)](assets/showcase/wan_fox_tacache.mp4) |
| At the edge of the sea | [![At the edge of the sea - original](assets/showcase/wan_lighthouse_original.gif)](assets/showcase/wan_lighthouse_original.mp4) | [![At the edge of the sea - tacache](assets/showcase/wan_lighthouse_tacache.gif)](assets/showcase/wan_lighthouse_tacache.mp4) |

Click a preview to open the video. [Prompts and settings](assets/showcase/prompts.json).

## Layout

```text
tacache/       shared algorithm and calibration
Wan2.1/        Wan sampling and model integration
models/        checkpoint directory
tests/         core regression tests
```

## Installation

Code release is in progress; the following commands apply to the upcoming release.

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

## Evaluation

Optional metrics:

```bash
python -m pip install -r requirements-eval.txt
```

Includes LPIPS, PyIQA and CLIP-IQA. [VBench](https://github.com/Vchitect/VBench) and [Q-Align](https://github.com/Q-Future/Q-Align) require separate environments.

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

## Citation

If you find this code helpful in your research or work, please consider citing:

```bibtex
@misc{liu2026acceleratingrectifiedflowmodels,
      title={Accelerating Rectified Flow Models via Trajectory-Aware Caching}, 
      author={Xiao Liu and Kai Liu and Naiyang Guan and Hongliang Lu and Zhixin Wang and Zhikai Chen and Renjing Pei and Yulun Zhang},
      year={2026},
      eprint={2605.16789},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.16789}, 
}

```

More backbones will be released.
