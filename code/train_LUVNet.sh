#!/bin/bash

SCRIPT_NAME="train_baseline.py"  # 실행할 Python 파일명

# 실험 변수 설정
MODEL_NAME="USVN"
ENCODER_NAME="densenet161_scratch"
SEED_NUM=234
DATA_VERSION="version_1"
FRAME_SIZE=256
LR=1e-6
TOTAL_EPOCHS=100
STOP_PATIENCE=10
TRAIN_LAYER="all"
MODEL_OUTPUT_CLASS=5
MODEL_TEST_RATE="0.2"
# BASE_PATH="/data2/hoon2/LUS_Dataset/csv_files/"
# SAVE_PATH="/data2/hoon2/Results/video_model2/"

# 트랩 설정: CTRL+C를 누르면 현재 실험을 중단하고 다음 실험으로 이동
trap "echo 'Experiment interrupted, skipping to next loop'; continue" SIGINT

# fold_num을 0부터 4까지 변경하면서 실험 실행
for FOLD_NUM in {1..4}
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
        --train_layer ${TRAIN_LAYER} \
        --model_output_class ${MODEL_OUTPUT_CLASS} \
        --model_test_rate ${MODEL_TEST_RATE} \
    
    echo "Completed training for fold ${FOLD_NUM}"
    echo "-----------------------------------"
done

echo "All experiments completed!"
