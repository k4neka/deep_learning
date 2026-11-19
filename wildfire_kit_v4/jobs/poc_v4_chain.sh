#!/bin/bash
# Strategy A, version 4: v3 reward + reversal penalty + a per-step cost for known fire that is being forgotten
# (see poc.py). Submits each variant as a chain of CPU jobs; each stage starts from the previous model and only
# runs if the previous stage finished.
#
# Run from wildfire_kit_v4 on the VSC login node:
#     bash jobs/poc_v4_chain.sh              # all variants
#     bash jobs/poc_v4_chain.sh mean ft      # only some variants
#
# Variants:
#   mean  cost = mean forgottenness of the whole fire map;   fire every 4 -> 2 -> 1, from scratch
#   max   cost = forgottenness of the most forgotten fire;    fire every 4 -> 2 -> 1, from scratch
#   ft    'mean', fine-tuned from the best v3 model (runs/v3_rev_fe2/model.zip); fire every 2 -> 1
set -e
cd "$(dirname "$0")/.."
VARIANTS=${@:-mean max ft}
COMMON="--preset v4 --new-rate 0.03 --eval-new-rate 0.01 --steps 2000000 --subproc"

submit() {   # submit NAME DEPENDENCY ARGS...  -> prints the job id, stops the script if sbatch refuses
    local name=$1 dep=$2; shift 2
    local d=""
    [ -n "$dep" ] && d="--dependency=afterok:$dep"
    local j
    j=$(sbatch --parsable $d --job-name="$name" jobs/exp_cpu.slurm train_poc.py "$@") || true
    j=${j%%;*}
    if ! [[ "$j" =~ ^[0-9]+$ ]]; then
        echo "sbatch refused job $name (see the message above); nothing more is submitted" >&2
        exit 1
    fi
    echo "$j"
}

for v in $VARIANTS; do
    case $v in
        mean) ARGS="--forget-mode mean"; STAGES="4 2 1"; prev_model="" ;;
        max)  ARGS="--forget-mode max";  STAGES="4 2 1"; prev_model="" ;;
        ft)   ARGS="--forget-mode mean"; STAGES="2 1";   prev_model="runs/v3_rev_fe2/model.zip"
              [ -f "$prev_model" ] || { echo "missing $prev_model" >&2; exit 1; } ;;
        *) echo "unknown variant $v" >&2; exit 1 ;;
    esac
    prev=""
    for fe in $STAGES; do
        out=runs/v4_${v}_fe${fe}
        init=""
        [ -n "$prev_model" ] && init="--init-from $prev_model"
        j=$(submit "v4_${v}_fe${fe}" "$prev" $ARGS --fire-every $fe $init $COMMON --out $out)
        echo "v4 $v fire every $fe: job $j${prev:+ (after $prev)}"
        prev=$j; prev_model=$out/model.zip
    done
done
echo "check with: squeue --clusters=wice -u \$USER"
