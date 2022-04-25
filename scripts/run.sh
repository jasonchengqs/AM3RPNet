# bash

GPU=1
MODE=train
CONFIG=./configs/slice_configs.yaml
# CONFIG=./configs/timeseries_configs.yaml
# CONFIG=./configs/dual_configs.yaml

python run.py \
    --gpu ${GPU} \
    --mode $MODE \
    --config-file ${CONFIG}
