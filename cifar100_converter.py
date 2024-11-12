import pickle
import numpy as np
import os
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_images(data, labels, coarse_labels, label_names, coarse_label_names, save_dir):
    for i, (img, label, coarse_label) in enumerate(zip(data, labels, coarse_labels)):
        img = img.reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img)
        coarse_class = coarse_label_names[coarse_label].decode('utf-8') 
        fine_class = label_names[label].decode('utf-8')
        class_dir = os.path.join(save_dir, coarse_class, fine_class)
        os.makedirs(class_dir, exist_ok=True)
        img.save(os.path.join(class_dir, f'{i}.png'))

# 데이터 로드
cifar100_dir = 'data/cifar100_python' 
train_data = unpickle(os.path.join(cifar100_dir, 'train'))
test_data = unpickle(os.path.join(cifar100_dir, 'test'))
meta = unpickle(os.path.join(cifar100_dir, 'meta'))

# 클래스 이름 로드
fine_label_names = meta[b'fine_label_names']
coarse_label_names = meta[b'coarse_label_names']

# 이미지 저장
save_images(train_data[b'data'], train_data[b'fine_labels'], train_data[b'coarse_labels'], 
            fine_label_names, coarse_label_names, 'data/cifar100_org/train')
save_images(test_data[b'data'], test_data[b'fine_labels'], test_data[b'coarse_labels'], 
            fine_label_names, coarse_label_names, 'data/cifar100_org/test')

print("데이터셋 변환 완료")
