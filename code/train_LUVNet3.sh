#!/bin/bash

SCRIPT_NAME="train_LUV_Net.py"  # 실행할 Python 파일명

# 실험 변수 설정
MODEL_NAME="LUV_Net"
ENCODER_NAME="densenet161_scratch"
SEED_NUM=234
DATA_VERSION="version_1"
FRAME_SIZE=256
BATCH_SIZE=4
LR=1e-6
TOTAL_EPOCHS=150
STOP_PATIENCE=20
POOLING_METHOD="attn_multilabel_conv"
NUM_HEADS=8
KERNEL_WIDTH=13
TRAIN_LAYER="all"
MODEL_OUTPUT_CLASS=5
MODEL_TEST_RATE="0.2"
# BASE_PATH="/data2/hoon2/LUS_Dataset/csv_files/"
# SAVE_PATH="/data2/hoon2/Results/video_model2/"

# 트랩 설정: CTRL+C를 누르면 현재 실험을 중단하고 다음 실험으로 이동
trap "echo 'Experiment interrupted, skipping to next loop'; continue" SIGINT

# fold_num을 0부터 4까지 변경하면서 실험 실행
for FOLD_NUM in {0..4}
do
    echo "Starting training for fold ${FOLD_NUM}"

    # Python 스크립트 실행
    python $SCRIPT_NAME \
        --model_name ${MODEL_NAME} \
        --encoder_name ${ENCODER_NAME} \
        --seed_num ${SEED_NUM} \
        --fold_num ${FOLD_NUM} \
        --version ${DATA_VERSION} \
        --frame_size ${FRAME_SIZE} \
        --lr ${LR} \
        --total_epochs ${TOTAL_EPOCHS} \
        --stop_patience_num ${STOP_PATIENCE} \
        --pooling_method ${POOLING_METHOD} \
        --num_heads ${NUM_HEADS} \
        --kernel_width ${KERNEL_WIDTH} \
        --train_layer ${TRAIN_LAYER} \
        --model_output_class ${MODEL_OUTPUT_CLASS} \
        --model_test_rate ${MODEL_TEST_RATE} \
        --base_path ${BASE_PATH} \
        --save_path ${SAVE_PATH}
    
    echo "Completed training for fold ${FOLD_NUM}"
    echo "-----------------------------------"
done

echo "All experiments completed!"
