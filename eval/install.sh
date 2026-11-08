#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# MMCV needs PyTorch to be installed before its extensions are built.
python -m pip install "setuptools<81" wheel numpy==1.26.4
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-build-isolation -r eval/requirements.txt

# Use the official metric sources with the shared dependencies above.
mkdir -p third_party
while read -r directory repository revision; do
  if [ ! -d "third_party/${directory}" ]; then
    source_dir=$(mktemp -d)
    curl -fL "https://codeload.github.com/${repository}/tar.gz/${revision}" \
      | tar -xz --strip-components=1 -C "${source_dir}"
    mv "${source_dir}" "third_party/${directory}"
  fi
done <<'SOURCES'
VBench Vchitect/VBench 1039fe66d7aa2f4b214003ca2d35f5011c48bfee
CLIP-IQA IceClear/CLIP-IQA 061fd3c7b1b40839959777897c764792760d693c
Q-Align Q-Future/Q-Align 14dfdffb3b970c0727d48938b8f3741a65acbd6e
SOURCES
