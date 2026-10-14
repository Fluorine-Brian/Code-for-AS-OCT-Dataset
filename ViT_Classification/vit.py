from __future__ import print_function

import glob
from itertools import chain
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import timm

torch.set_num_threads(4)
torch.set_num_interop_threads(4)

batch_size = 64
epochs = 100
lr = 3e-4
gamma = 0.7
seed = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

device = 'cuda'

# AS-OCT Dataset Path
data_root = '/mnt/dataset4/fuxy/test/dataset/asoct_dataset_cleaned_0112'

# Four Catagories
class_names = ['Cataract', 'Normal', 'Glaucoma', 'Glaucoma_Cataract']
num_classes = 4

class_to_idx = {name: idx for idx, name in enumerate(class_names)}
all_images = []
labels = []

for class_name in class_names:
    class_dir = os.path.join(data_root, class_name)
    class_dir = os.path.join(class_dir, 'Original Images')
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        images = glob.glob(os.path.join(class_dir, ext))
        all_images.extend(images)
        labels.extend([class_name] * len(images))

from collections import Counter
label_counts = Counter(labels)
for class_name, count in label_counts.items():
    print(f"{class_name}: {count}")

train_list = all_images

train_list, temp_list, train_labels, temp_labels = train_test_split(
    all_images, 
    labels, 
    test_size=0.30, 
    random_state=seed, 
    stratify=labels
)

valid_list, test_list, valid_labels, test_labels = train_test_split(
    temp_list, 
    temp_labels, 
    test_size=0.50, 
    random_state=seed, 
    stratify=temp_labels
)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


test_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


class ASOCTDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')
        img_transformed = self.transform(img)

        label = -1
        for name, idx_val in class_to_idx.items():
            if name in img_path:
                label = idx_val
                break

        return img_transformed, label

train_data = ASOCTDataset(train_list, transform=train_transforms)
valid_data = ASOCTDataset(valid_list, transform=test_transforms)
test_data = ASOCTDataset(test_list, transform=test_transforms)

train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True, 
    num_workers=2)
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True, 
    num_workers=2)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True, 
    num_workers=2)

model_name = 'vit_small_patch16_224'
model = timm.create_model(model_name, pretrained=True, num_classes=4)
model = model.to(device)


# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

best_val_acc = 0.0

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    if epoch_val_accuracy > best_val_acc:
        best_val_acc = epoch_val_accuracy
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"Epoch {epoch+1}: New best model saved with val_acc: {epoch_val_accuracy:.4f}")

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )

print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")

model.load_state_dict(torch.load('best_model.pt'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    test_accuracy = 0
    for data, label in tqdm(test_loader, desc="Testing"):
        data = data.to(device)
        label = label.to(device)
        
        output = model(data)
        preds = output.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())
        
        acc = (preds == label).float().mean()
        test_accuracy += acc / len(test_loader)

print(f"\n{'='*50}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"{'='*50}\n")

all_possible_labels = list(range(len(class_names)))

print(classification_report(
    all_labels, 
    all_preds, 
    labels=all_possible_labels,
    target_names=class_names,
    digits=4
))