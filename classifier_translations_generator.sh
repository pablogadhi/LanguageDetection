#!/bin/bash

inputLangs=("es" "fr")
outputLangs=("es" "fr" "en" "de")

for input in ${inputLangs[@]}; do
    for output in ${outputLangs[@]}; do
        if [ "$input" != "$output" ]; then
            python tools/google_translator.py --source "data/aligned/classifier/${input}.txt" --output "data/translated/${input}_${output}.txt" --tgt "${output}"
        fi
    done
done
