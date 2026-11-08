# Evaluation setup

All metrics use one `tacache-eval` environment. Run from the repository root:

```bash
conda create -n tacache-eval python=3.10 -y
conda activate tacache-eval
bash eval/install.sh
```

The installer uses the shared dependencies in `eval/requirements.txt` and downloads fixed versions of the official metric sources into `third_party/`. MMCV may require a CUDA toolkit to compile.

See [REPRODUCE.md](../REPRODUCE.md#evaluation) for evaluation commands.
