#!/bin/bash

DATASETS=(
    'None'                # dummy
    'aquatic_mammals'
    'fish'
    'flowers'
    'food_containers'
    'fruit_and_vegetables'
    'household_electrical_devices'
    'household_furniture'
    'insects'
    'large_carnivores'
    'large_man-made_outdoor_things'
    'large_natural_outdoor_scenes'
    'large_omnivores_and_herbivores'
    'medium_mammals'
    'non-insect_invertebrates'
    'people'
    'reptiles'
    'small_mammals'
    'trees'
    'vehicles_1'
    'vehicles_2'
)

GPU_ID=0
ARCH='lenet5'
FINETUNE_EPOCHS=100
seed=2

####################
##### Baseline #####
####################

for TASK_ID in {1..20}
do
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 packnet_cifar100_main_normal.py \
        --arch $ARCH \
        --dataset ${DATASETS[TASK_ID]} --num_classes 5 \
        --lr 1e-2 \
        --weight_decay 4e-5 \
        --save_folder checkpoints_${ARCH}/baseline_scratch/$ARCH/${DATASETS[TASK_ID]} \
        --epochs $FINETUNE_EPOCHS \
        --mode finetune \
        --logfile logs_${ARCH}/baseline_cifar100_acc_scratch.txt \
        --seed $seed
done
# 실행하고 log_lenet5/baseline_cifar100_acc_scratch.txt 확인
# 자세하게 log 확인해보고 싶다면 open_checkpoint.py 실행
