# bash 

# windows
DATA_DIR="../r3DiM_Benchmark"
MODE=run
GCODE_DIR=${DATA_DIR}/GCODE_files
GCODE_SAVE_DIR="./results/preprocess/timeseries"
STL_DIR=${DATA_DIR}/STL_files
STL_SAVE_DIR="./results/preprocess/voxel"
LAYER_ENERGE_DIR=${DATA_DIR}/LayerData

python preprocess.py \
    --mode $MODE \
    --gcode-dir $GCODE_DIR \
    --gcode-save-dir $GCODE_SAVE_DIR \
    --stl-dir $STL_DIR \
    --stl-save-dir $STL_SAVE_DIR \
    --layer-resource-dir ${LAYER_ENERGE_DIR} \
    --save-slices