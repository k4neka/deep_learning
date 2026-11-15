#!/bin/bash
# Strategy A, version 3: submits the curriculum (fire every 4 -> 2 -> 1 steps) for each variant as a chain of
# CPU jobs; each stage starts from the previous model and only runs if the previous stage finished.
# Also re-evaluates the version 2 models with the new metrics (fire age, revisit gap, dwell, reversals).
#
# Run from wildfire_kit_v4 on the VSC login node:
#     bash jobs/poc_v3_chain.sh            # all variants + v2 re-evaluation
#     bash jobs/poc_v3_chain.sh age rev    # only some variants
#
# Variants:
#   age   v3 reward: IoU (0.3) + corrections + age of the refreshed information
#   rev   age + penalty 0.1 for reversing the previous move (against back-and-forth jitter)
#   goto  age + 'fly to one of 8 x 8 sectors' actions (up to 8 steps per decision; --steps counts decisions)
#   v2    no training: evaluates runs/v2_fe{4,2,1}/model.zip with the new metrics (reference)
set -e
cd "$(dirname "$0")/.."
VARIANTS=${@:-age rev goto v2}
COMMON="--new-rate 0.03 --eval-new-rate 0.01 --subproc"

submit() {   # submit NAME DEPENDENCY ARGS...  -> prints the job id
    local name=$1 dep=$2; shift 2
    local d=""
    [ -n "$dep" ] && d="--dependency=afterok:$dep"
    local j
    j=$(sbatch --parsable $d --job-name="$name" jobs/exp_cpu.slurm train_poc.py "$@")
    echo "${j%%;*}"
}

for v in $VARIANTS; do
    case $v in
        age)  ARGS="--preset v3";                     STEPS=2000000 ;;
        rev)  ARGS="--preset v3 --w-rev 0.1";         STEPS=2000000 ;;
        goto) ARGS="--preset v3 --action-mode goto";  STEPS=300000 ;;
        v2)
            for fe in 4 2 1; do
                j=$(submit "v2_fe${fe}_eval" "" --steps 0 --init-from runs/v2_fe${fe}/model.zip --stale --exp-floor 0.3 \
                    --fire-every $fe $COMMON --out runs/v2_fe${fe}_eval)
                echo "v2 fire every $fe: re-evaluation job $j"
            done
            continue ;;
        *) echo "unknown variant $v"; exit 1 ;;
    esac
    prev=""; prev_model=""
    for fe in 4 2 1; do
        out=runs/v3_${v}_fe${fe}
        init=""
        [ -n "$prev_model" ] && init="--init-from $prev_model"
        j=$(submit "v3_${v}_fe${fe}" "$prev" $ARGS --fire-every $fe --steps $STEPS $init $COMMON --out $out)
        echo "v3 $v fire every $fe: job $j${prev:+ (after $prev)}"
        prev=$j; prev_model=$out/model.zip
    done
done
echo "check with: squeue --clusters=wice -u \$USER"
