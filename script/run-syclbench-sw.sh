#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]:-$0}")")

export LD_LIBRARY_PATH=/home/export/online1/mdt00/shisuan/swyjs/oyyc/lib:$LD_LIBRARY_PATH

out_args="--output=sycl-bench.csv"
DEFAULT_ARGS="--size=1024"

declare -A ARGS=(
    [scalar_prod]="--size=1024"
    [vec_add]="--size=1024"
)

for file in ${SCRIPT_DIR}/benchmarks/*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        args="${ARGS[$filename]:-$DEFAULT_ARGS}"

        cmd=(bsub -J "$filename" -q q_share -b -m 1 -n 1 -cgsp 64 -share_size 13000 -priv_size 16 -host_stack 1024 -cache_size 128 "$file" $args $out_args)
        echo "+ ${cmd[*]}"
        "${cmd[@]}"

        if [ $? -eq 0 ]; then
            echo "=== Successfully submitted: $filename ==="
        else
            echo "=== Failed to submit: $filename ==="
        fi
    fi
done
