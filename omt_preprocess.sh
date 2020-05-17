#!/bin/bash

data_dir="./data/*"

for entry in $data_dir; do
    lang_pair=$(echo $entry | cut -d "/" -f 3)
    lang_0=$(echo $lang_pair | cut -d "-" -f 1)
    lang_1=$(echo $lang_pair | cut -d "-" -f 2)

    lang_file_0="${lang_pair}_${lang_0}.txt"
    lang_file_1="${lang_pair}_${lang_1}.txt"
    lang_val_0="${lang_pair}_${lang_0}_val.txt"
    lang_val_1="${lang_pair}_${lang_1}_val.txt"

    subdir="./data/${lang_pair}"
    onmt_preprocess -train_src "${subdir}/${lang_file_0}" \
        -train_tgt "${subdir}/${lang_file_1}" \
        -valid_src "${subdir}/${lang_val_0}" \
        -valid_tgt "${subdir}/${lang_val_1}" \
        -save_data "${subdir}/${lang_pair}" \
        -num_threads 8 \
        -overwrite
done
