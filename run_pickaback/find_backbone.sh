
GPU_ID=0
RESULT_FILE="find_backbone_result.csv"

if [ -f $RESULT_FILE ]; then
    rm $RESULT_FILE
fi

echo "target_id,selected_backbone" > $RESULT_FILE

for TARGET_ID in {1..20}
do
    CUDA_VISIBLE_DEVICES=$GPU_ID TARGET_ID=$TARGET_ID python3 pickaback_cifar100.py
done