import torch

#원하는 task명으로 file_path 수정해가면서 찍어봐야 함
file_path = 'checkpoints_lenet5/baseline_scratch/lenet5/household_furniture/checkpoint-100.pth.tar'

checkpoint = torch.load(file_path, weights_only=True)

print(checkpoint.keys())  # 키 목록 확인

if 'model_state_dict' in checkpoint:
    print(checkpoint['model_state_dict'].keys())

# 각 항목의 세부 내용 확인 (필요한 경우 더 추가 탐색)
print("Dataset History:", checkpoint['dataset_history'])
print("Dataset to Number of Classes:", checkpoint['dataset2num_classes'])
print("Masks:", checkpoint['masks'])  # 필요한 경우 특정 레이어별로 출력
print("Shared Layer Info:", checkpoint['shared_layer_info'])
