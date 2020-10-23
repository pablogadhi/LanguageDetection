#!/bin/bash

inputDir=$1
outputDir=$2
script=$3

inputLangs=("es" "fr" "en" "de")
outputLangs=("es" "fr" "en" "de")

for input in ${inputLangs[@]}; do
    for output in ${outputLangs[@]}; do
        if [ "$input" != "$output" ]; then
            echo "Translating from ${input} to ${output}..."
            if [ $script == "google" ]; then
                python tools/google_translator.py --source "${inputDir}/${input}.txt" --output "${outputDir}/${input}_${output}.txt" --tgt "${output}"
            else
                python tools/remote_translate.py --src_file "${inputDir}/${input}.txt" --output "${outputDir}/${input}_${output}.txt" --src "${input}" --tgt "${output}"
            fi
        fi
    done
done
