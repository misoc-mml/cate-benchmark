ITER=10
simple=("dummy" "lr" "lr-ipw" "ridge" "ridge-ipw" "lasso" "kr" "kr-ipw" "dt" "dt-ipw" "et" "et-ipw" "lgbm" "lgbm-ipw" "cb" "cb-ipw" "cf")
meta=("dml" "dr" "tl" "xl")
base=("lr" "ridge" "lasso" "kr" "dt" "et" "lgbm" "cb")

# Standalone estimators
for MODEL in ${simple[@]}
do
    echo $MODEL
    python -W ignore ../main.py --data_path ../datasets/JOBS --dtype jobs --iter $ITER -o ../results/jobs_${MODEL} --sr --tbv --em $MODEL
done

# Meta models
for MODEL in ${meta[@]}
do
    for BASE_MODEL in ${base[@]}
    do
        echo ${MODEL}_${BASE_MODEL}
        python -W ignore ../main.py --data_path ../datasets/JOBS --dtype jobs --iter $ITER -o ../results/jobs_${MODEL}-${BASE_MODEL} --sr --tbv --em $MODEL --ebm $BASE_MODEL
    done
done

# Post-processing the results
python ../results/process.py --data_path ../results --dtype jobs -o ../results --sm ${simple[@]} --mm ${meta[@]} --bm ${base[@]} --show