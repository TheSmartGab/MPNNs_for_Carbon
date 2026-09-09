#!/bin/bash

target_name=$1

# Read all matching files into an array safely (handles spaces)
readarray -d '' files < <(find . -type f -name "*${target_name}*" -print0)

# Debug: print found files
printf '%s\n' "${files[@]}"

for i in "${files[@]}"; do
    evince "$i" &
    pid=$!
    sleep 5
    kill "$pid"
done

