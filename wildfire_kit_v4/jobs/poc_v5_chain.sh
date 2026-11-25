#!/bin/bash
# Strategy A, version 5: v4 with the w1 weights (forget cost 1.0) + ALTITUDE (up / down actions; low = small
# view, perfect sensor; high = large view, noisy sensor) + gradual HEAT DAMAGE when flying low over intense fire
# (see poc.py). Each seed is a chain of CPU jobs, fire every 4 -> 2 -> 1 steps, from scratch (the observation has
# two more channels than v4, so v4 models cannot be reused). A stage only runs if the previous one finished.
#
# Run from wildfire_kit_v4 on the VSC login node:
#     bash jobs/poc_v5_chain.sh          # seeds 0 and 1 (6 jobs, about 1 h each)
#     bash jobs/poc_v5_chain.sh s0       # only seed 0
set -e
cd "$(dirname "$0")/.."
VARIANTS=${@:-s0 s1}
COMMON="--preset v5 --new-rate 0.03 --eval-new-rate 0.01 --steps 2000000 --subproc"

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
        s0) seed=0 ;;
        s1) seed=1 ;;
        s2) seed=2 ;;
        *) echo "unknown variant $v (use s0, s1 or s2)" >&2; exit 1 ;;
    esac
    prev=""; prev_model=""
    for fe in 4 2 1; do
        out=runs/v5_s${seed}_fe${fe}
        init=""
        [ -n "$prev_model" ] && init="--init-from $prev_model"
        j=$(submit "v5_s${seed}_fe${fe}" "$prev" --seed $seed --fire-every $fe $init $COMMON --out $out)
        echo "v5 seed $seed fire every $fe: job $j${prev:+ (after $prev)}"
        prev=$j; prev_model=$out/model.zip
    done
done
echo "check with: squeue --clusters=wice -u \$USER"
