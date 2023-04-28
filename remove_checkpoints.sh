#!/bin/bash

for d in trained_models/*; do
  if [[ "$d" == *"contdist"* ]]; then
    echo "$d"
    find "$d" -type f ! -size 0c | parallel -X --progress truncate -s0
    rm -rf "$d"
  fi


done
