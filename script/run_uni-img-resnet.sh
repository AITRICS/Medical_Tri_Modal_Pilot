# modify below working directories..
IMG_DIR=/mnt/aitrics_ext/ext01/alex/multi_modal/
TRAIN_DIR=/nfs/thena/shared/multi_modal/training_data_0725/mimic_icu/train
TEST_DIR=/nfs/thena/shared/multi_modal/training_data_0725/mimic_icu/test
RESULT_DIR=/mnt/aitrics_ext/ext01/alex/multi_modal/uni_img/resnet34

CUDA_VISIBLE_DEVICES=6 nohup python ./2_train.py \
    --train-data-path $TRAIN_DIR --test-data-path $TEST_DIR --image-data-path $IMG_DIR\
    --project-name MOR72_ResNet34_1e-4 --model resnet --resnet_layer 34 \
    --input-types img --output-type mortality \
    --predict-type binary --prediction-range 72 \
    --optim adamw --epoch 155 --batch-size 32 --cross-fold-val True \
    --lr-scheduler CosineAnnealing  --lr-init 1e-4 \
    > $RESULT_DIR/exp01/MOR72_ResNet34_1e-4.out &
