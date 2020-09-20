#!/bin/bash

inputDir=$1
outputDir=$2

inputLangs=("es" "fr")
outputLangs=("es" "fr" "en" "de")

for input in ${inputLangs[@]}; do
    for output in ${outputLangs[@]}; do
        if [ "$input" != "$output" ]; then
            python tools/google_translator.py --source "${inputDir}/${input}.txt" --output "${outputDir}/${input}_${output}.txt" --tgt "${output}"
        fi
    done
done
