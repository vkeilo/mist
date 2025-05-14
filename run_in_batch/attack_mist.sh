# export EXPERIMENT_NAME=$data_id
export EXPERIMENT_NAME=$EXPERIMENT_NAME
export MODEL_PATH=$model_path
export CLEAN_TRAIN_DIR="$data_path/$dataset_name/$data_id/set_A" 
export CLEAN_ADV_DIR="$data_path/$dataset_name/$data_id/set_B"
export CLEAN_REF="$data_path/$dataset_name/$data_id/set_C"
# export OUTPUT_DIR="outputs/simac/$dataset_name/$EXPERIMENT_NAME"
export OUTPUT_DIR="outputs/$EXPERIMENT_NAME/$wandb_run_name"
export CLASS_DIR=$class_dir


# ------------------------- Train ASPL on set B -------------------------
mkdir -p $OUTPUT_DIR
rm -r $OUTPUT_DIR/* 2>/dev/null || true
cp -r $CLEAN_REF $OUTPUT_DIR/image_clean_ref
cp -r $CLEAN_ADV_DIR $OUTPUT_DIR/image_before_addding_noise
# mist deviation correction
r=$((r - 1))
mist_cmd="""python mist_v3.py \
--model_path=$MODEL_PATH  \
--input_dir_path=$CLEAN_ADV_DIR \
--output_dir=$OUTPUT_DIR \
--epsilon=$r \
--concept_prompt='$concept_prompt' \
--model_config=$model_config \
--mode=$mode \
--rate=$rate \
--block_num=$block_num \
--input_size=$input_size \
--steps=$attack_steps \
"""
echo $mist_cmd
eval $mist_cmd




