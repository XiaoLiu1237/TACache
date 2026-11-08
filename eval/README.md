# Evaluation setup

Run from the TACache repository root. PSNR, SSIM and LPIPS use the `tacache` environment. The other metrics use separate environments for their upstream dependencies.

## PSNR / SSIM / LPIPS

```bash
conda activate tacache
python -m pip install -r eval/requirements.txt
```

## VBench / Q-Align

Run once for each metric you need, setting `metric` to `vbench` or `qalign`:

```bash
metric=vbench
conda create -n "tacache-${metric}" python=3.10 -y
conda activate "tacache-${metric}"
python -m pip install -r "eval/requirements-${metric}.txt"
```

## CLIP-IQA

Install PyTorch before MMCV, then install the official CLIP-IQA source:

```bash
conda create -n tacache-clipiqa python=3.10 -y
conda activate tacache-clipiqa
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/IceClear/CLIP-IQA.git third_party/CLIP-IQA
git -C third_party/CLIP-IQA checkout 061fd3c7b1b40839959777897c764792760d693c
python -m pip install -r eval/requirements-clipiqa.txt
```

MMCV may require a CUDA toolkit to compile. See [REPRODUCE.md](../REPRODUCE.md#evaluation) for evaluation commands.
