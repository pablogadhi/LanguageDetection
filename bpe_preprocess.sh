#!/bin/bash

python tools/tokenize.py --source data/aligned/train --output data/aligned/train/rawtok/ --out_ext .txt

python tools/learn_bpe.py --source data/aligned/train/rawtok --output models/bpe/bpe_model --symbols 30000

python tools/tokenize.py --source data/aligned/train --output data/tokenized/train --out_ext .txt --joiner_annotate --bpe_model models/bpe/bpe_mode
python tools/tokenize.py --source data/aligned/validation --output data/tokenized/validation --out_ext .txt --joiner_annotate --bpe_model models/bpe/bpe_model
python tools/tokenize.py --source data/aligned/test --output data/tokenized/test --out_ext .txt --joiner_annotate --bpe_model models/bpe/bpe_model
