
GPU_ID=0
for TARGET_ID in {1..20}
do
    CUDA_VISIBLE_DEVICES=$GPU_ID TARGET_ID=$TARGET_ID python3 pickaback_cifar100.py
done