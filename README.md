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

## Abstract

> Diffusion and rectified flow (RF) models generate high-fidelity images and videos, but their iterative velocity-field evaluations are computationally expensive. Existing caching methods accelerate sampling by skipping timesteps, yet their coarse approximations introduce accumulated errors over long skip intervals and degrade quality under aggressive acceleration. We propose **TACache (Trajectory-Aware Cache)**, a training-free acceleration framework following a **skip-then-compensate** paradigm. TACache performs an orthogonal decomposition of discrete velocity acceleration along the RF trajectory into a parallel component and an orthogonal residual, isolating the magnitude and directional sources of per-step approximation error. The framework operates in two stages: *offline*, cumulative variation thresholds on the magnitude and direction indicators yield the skip schedule and bound how far each skip interval may extend; *online*, at each skipped step the offline statistics are combined with the sample's historical orthogonal direction to reconstruct the skipped velocity without additional model evaluations. Main comparisons on BAGEL, FLUX.1-dev, and Wan2.1-1.3B show that TACache achieves up to 4.11× speedup on text-to-image generation and 2.11× speedup on text-to-video generation, with consistent improvements over prior cache-based methods on all reference-based fidelity metrics.

---

## ✨ Pipeline
<p align="center">
  <img src="fig/overview.png" width="92%">
</p>


---

## 🚀 TODO
- [ ] Release the calibration dataset.
- [ ] Release code.

---

## 🔗 Contents
1. [code](#code)
2. [Results](#results)
3. [Citation](#citation)

---

## 📦 Results
<p align="center">
  <img src="fig/cover.pdf" width="92%">
</p>
<p align="center">
  <img src="fig/bagel.png" width="92%">
</p>
<p align="center">
  <img src="fig/flux.png" width="92%">
</p>
<p align="center">
  <img src="fig/wan.png" width="92%">
</p>

## 📎 Citation

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
