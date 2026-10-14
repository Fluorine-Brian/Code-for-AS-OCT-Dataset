import os
import torch
torch.set_num_threads(4)

import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# AS-OCT Data Path
DATA_ROOT = '/mnt/dataset4/fuxy/test/dataset/asoct_dataset_cleaned_0112'
MODEL_SAVE_DIR = 'pth/asoct'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

CLASS_NAMES = ['Cataract', 'Glaucoma', 'Glaucoma_Cataract', 'Normal']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class ASOCTDataset(Dataset):

    def __init__(self, img_paths, class_names, class_to_idx, transform=None):
        self.img_paths = img_paths
        self.class_names = class_names
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.labels = []

        for img_path in self.img_paths:
            label = self._get_label_from_path(img_path)
            self.labels.append(label)

    def _get_label_from_path(self, img_path):

        if 'Glaucoma_Cataract' in img_path:
            return self.class_to_idx['Glaucoma_Cataract']
        elif 'Cataract' in img_path:
            return self.class_to_idx['Cataract']
        elif 'Glaucoma' in img_path:
            return self.class_to_idx['Glaucoma']
        elif 'Normal' in img_path:
            return self.class_to_idx['Normal']
        else:
            raise ValueError(f"Cannot determine class for path: {img_path}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def collect_all_images(data_root):
    all_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        pattern = os.path.join(data_root, '**', ext)
        all_images.extend(glob.glob(pattern, recursive=True))
    return sorted(all_images)

all_image_paths = collect_all_images(DATA_ROOT)

class_counts = {name: 0 for name in CLASS_NAMES}
for path in all_image_paths:
    if 'Glaucoma_Cataract' in path:
        class_counts['Glaucoma_Cataract'] += 1
    elif 'Cataract' in path:
        class_counts['Cataract'] += 1
    elif 'Glaucoma' in path:
        class_counts['Glaucoma'] += 1
    elif 'Normal' in path:
        class_counts['Normal'] += 1


random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

indices = np.random.permutation(len(all_image_paths))
shuffled_paths = [all_image_paths[i] for i in indices]

total_size = len(shuffled_paths)
train_size = int(0.70 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size 

train_paths = shuffled_paths[:train_size]
val_paths = shuffled_paths[train_size:train_size + val_size]
test_paths = shuffled_paths[train_size + val_size:]

print(f"  Train: {len(train_paths)} images ({len(train_paths)/total_size*100:.1f}%)")
print(f"  Val:   {len(val_paths)} images ({len(val_paths)/total_size*100:.1f}%)")
print(f"  Test:  {len(test_paths)} images ({len(test_paths)/total_size*100:.1f}%)")

test_paths_file = os.path.join(MODEL_SAVE_DIR, 'test_paths.txt')
with open(test_paths_file, 'w') as f:
    for path in test_paths:
        f.write(path + '\n')

train_dataset = ASOCTDataset(train_paths, CLASS_NAMES, CLASS_TO_IDX, transform=train_transforms)
val_dataset = ASOCTDataset(val_paths, CLASS_NAMES, CLASS_TO_IDX, transform=val_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(2048, NUM_CLASSES)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 50
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_acc = 100 * train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch + 1}/{num_epochs}] | "
          f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | "
          f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
    
    # Early Stopping Check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_path = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f"  -> Best model saved (val_loss: {best_val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    scheduler.step()

final_model_path = os.path.join(MODEL_SAVE_DIR, 'final_model.pth')
torch.save(model.state_dict(), final_model_path)
