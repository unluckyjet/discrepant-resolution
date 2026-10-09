#!/bin/sh
# Fetch CoNLL-2003 test (eng.testb, via the mirror used by Reiss et al.'s tooling), CoNLL++ (Wang et al. 2019)
# and Reiss et al.'s (2020) released audit at a pinned commit. CoNLL-2003 is licensed for research use only.
set -e
cd "$(dirname "$0")/../.."
mkdir -p data/raw/conll && cd data/raw/conll
curl -sSL -o eng.testb https://github.com/patverga/torch-ner-nlp-from-scratch/raw/master/data/conll2003/eng.testb
curl -sSL -o conllpp_test.txt https://raw.githubusercontent.com/ZihanWangKi/CrossWeigh/master/data/conllpp_test.txt
if [ ! -d reiss ]; then
  git clone -q https://github.com/CODAIT/Identifying-Incorrect-Labels-In-CoNLL-2003.git reiss
  git -C reiss checkout -q b1eccd8
fi
