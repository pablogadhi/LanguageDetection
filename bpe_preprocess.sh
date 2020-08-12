#!/bin/bash

python tools/tokenize.py --source data/aligned/train --out_ext .txt --subdir rawtok

python tools/learn_bpe.py --source data/aligned/train/rawtok --output data/aligned/bpe_model --share_src

python tools/tokenize.py --source data/aligned/train --out_ext .txt --subdir tokenized --bpe_model data/aligned/bpe_model --joiner_annotate
python tools/tokenize.py --source data/aligned/validation --out_ext .txt --subdir tokenized --bpe_model data/aligned/bpe_model --joiner_annotate
python tools/tokenize.py --source data/aligned/test --out_ext .txt --subdir tokenized --bpe_model data/aligned/bpe_model --joiner_annotate
