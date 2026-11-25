#!/bin/bash
# Strategy A v4, follow-up runs after the first v4 results (24 Nov): a stronger cost for forgotten fire, longer
# training at full fire speed, and extra seeds of the best variant (v4 mean) to measure the spread between runs.
# Each chain submits CPU jobs; a stage only runs if the previous one finished, starting from its model.
# Evaluation missions are the same as before (seeds 5000-5029), so all results are paired and comparable.
#
# Run from wildfire_kit_v4 on the VSC login node:
#     bash jobs/poc_v4_tune.sh w1s1 w1s2      # the variants to run (names below; at least one is required)
#
# Variants:
#   w1    v4 mean with the cost of forgotten fire doubled (0.5 -> 1.0)          fire every 4 -> 2 -> 1
#   t20   v4 mean with fire counted as forgotten after 20 steps instead of 30  fire every 4 -> 2 -> 1
#   long  runs/v4_mean_fe1 trained 4 M more steps at full fire speed           fire every 1
#   s1    v4 mean again with training seed 1                                    fire every 4 -> 2 -> 1
#   s2    v4 mean again with training seed 2                                    fire every 4 -> 2 -> 1
#   w1s1  w1 again with training seed 1 (to confirm w1, added 25 Nov)          fire every 4 -> 2 -> 1
#   w1s2  w1 again with training seed 2                                         fire every 4 -> 2 -> 1
set -e
cd "$(dirname "$0")/.."
VARIANTS="$*"
[ -n "$VARIANTS" ] || { echo "usage: bash jobs/poc_v4_tune.sh VARIANT... (w1 t20 long s1 s2 w1s1 w1s2)" >&2; exit 1; }
COMMON="--preset v4 --forget-mode mean --new-rate 0.03 --eval-new-rate 0.01 --subproc"

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

chain() {    # chain NAME "EXTRA ARGS": fire every 4 -> 2 -> 1, 2 M steps per stage
    local name=$1 extra=$2 prev="" prev_model="" fe out init j
    for fe in 4 2 1; do
        out=runs/v4_${name}_fe${fe}
        init=""
        [ -n "$prev_model" ] && init="--init-from $prev_model"
        j=$(submit "v4_${name}_fe${fe}" "$prev" $extra --fire-every $fe --steps 2000000 $init $COMMON --out $out)
        echo "v4 $name fire every $fe: job $j${prev:+ (after $prev)}"
        prev=$j; prev_model=$out/model.zip
    done
}

for v in $VARIANTS; do
    case $v in
        w1)   chain w1  "--w-forget 1.0" ;;
        t20)  chain t20 "--t-forget 20" ;;
        s1)   chain mean_s1 "--seed 1" ;;
        s2)   chain mean_s2 "--seed 2" ;;
        w1s1) chain w1_s1 "--w-forget 1.0 --seed 1" ;;
        w1s2) chain w1_s2 "--w-forget 1.0 --seed 2" ;;
        long)
            [ -f runs/v4_mean_fe1/model.zip ] || { echo "missing runs/v4_mean_fe1/model.zip" >&2; exit 1; }
            j=$(submit "v4_mean_fe1_long" "" --fire-every 1 --steps 4000000 --init-from runs/v4_mean_fe1/model.zip \
                $COMMON --out runs/v4_mean_fe1_long)
            echo "v4 mean, 4 M more steps at full fire speed: job $j" ;;
        *) echo "unknown variant $v" >&2; exit 1 ;;
    esac
done
echo "check with: squeue --clusters=wice -u \$USER"
