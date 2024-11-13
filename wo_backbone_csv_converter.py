import os
import re
import pandas as pd

# 기본 경로 설정
base_dir = 'checkpoints_lenet5/CPG_single_scratch_woexp/lenet5'
datasets = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
    'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
    'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
    'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals', 'trees',
    'vehicles_1', 'vehicles_2'
]

# 패턴 정의
train_pattern = r"In train\(\)-> Train Ep. #(\d+) loss: ([\d.]+), accuracy: ([\d.]+), lr: ([\d.]+), sparsity: ([\d.]+)"
val_pattern = r"In validate\(\)-> Val Ep. #(\d+) loss: ([\d.]+), accuracy: ([\d.]+), sparsity: ([\d.]+)"

for dataset in datasets:
    log_path = os.path.join(base_dir, dataset, 'train.log')
    csv_path = os.path.join(base_dir, dataset, 'train.csv')
    
    if not os.path.exists(log_path):
        print(f"Log file not found for dataset: {dataset}")
        continue

    # 데이터를 담을 리스트 초기화
    data = []

    # 로그 파일에서 데이터 추출
    with open(log_path, 'r') as f:
        for line in f:
            train_match = re.search(train_pattern, line)
            val_match = re.search(val_pattern, line)
            
            if train_match:
                # Train 데이터 추출
                epoch = int(train_match.group(1))
                loss = float(train_match.group(2))
                accuracy = float(train_match.group(3))
                lr = float(train_match.group(4))
                sparsity = float(train_match.group(5))
                data.append([epoch, 'train', lr, accuracy, loss, sparsity])
                
            elif val_match:
                # Validation 데이터 추출
                epoch = int(val_match.group(1))
                loss = float(val_match.group(2))
                accuracy = float(val_match.group(3))
                sparsity = float(val_match.group(4))
                data.append([epoch, 'val', None, accuracy, loss, sparsity])

    # DataFrame으로 변환하여 CSV 파일로 저장
    df = pd.DataFrame(data, columns=['Epoch', 'Mode', 'Learning Rate', 'Accuracy', 'Loss', 'Sparsity'])
    df.to_csv(csv_path, index=False)
    print(f"Converted {log_path} to {csv_path}")
