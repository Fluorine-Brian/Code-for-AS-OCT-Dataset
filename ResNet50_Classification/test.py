import os
import torch
torch.set_num_threads(4)

from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data and model paths
MODEL_SAVE_DIR = 'pth/asoct'
RESULTS_DIR = 'results/asoct'
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES = ['Cataract', 'Glaucoma', 'Glaucoma_Cataract', 'Normal']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASS_NAMES)

test_transforms = transforms.Compose([
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


test_paths_file = os.path.join(MODEL_SAVE_DIR, 'test_paths.txt')

with open(test_paths_file, 'r') as f:
    test_paths = [line.strip() for line in f if line.strip()]

test_class_counts = {name: 0 for name in CLASS_NAMES}
for path in test_paths:
    if 'Glaucoma_Cataract' in path:
        test_class_counts['Glaucoma_Cataract'] += 1
    elif 'Cataract' in path:
        test_class_counts['Cataract'] += 1
    elif 'Glaucoma' in path:
        test_class_counts['Glaucoma'] += 1
    elif 'Normal' in path:
        test_class_counts['Normal'] += 1

test_dataset = ASOCTDataset(test_paths, CLASS_NAMES, CLASS_TO_IDX, transform=test_transforms)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(2048, NUM_CLASSES)
)

model_path = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')
if not os.path.exists(model_path):
    model_path = os.path.join(MODEL_SAVE_DIR, 'final_model.pth')

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)
all_probabilities = np.array(all_probabilities)

accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')

print(f"Total test samples: {len(all_labels)}")
print(f"Overall Accuracy   : {accuracy * 100:.2f}%")
print(f"Weighted Precision : {precision * 100:.2f}%")
print(f"Weighted Recall    : {recall * 100:.2f}%")
print(f"Weighted F1 Score  : {f1 * 100:.2f}%")
report = classification_report(
    all_labels,
    all_predictions,
    target_names=CLASS_NAMES,
    digits=4
)
print(report)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, annot_kws={"size": 12})

    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.title("Confusion Matrix - AS-OCT Classification", fontsize=16)

    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_pr_roc_curves(y_true, y_scores, n_classes, class_names, save_dir=None):
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        precision_vals, recall_vals, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        plt.plot(recall_vals, precision_vals, label=f'{class_names[i]}', linewidth=2)

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - AS-OCT Classification', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir:
        pr_path = os.path.join(save_dir, "precision_recall_curve.png")
        plt.savefig(pr_path, dpi=150)
        print(f"Precision-Recall curve saved to: {pr_path}")
    else:
        plt.show()
    plt.close()

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.3f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - AS-OCT Classification', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir:
        roc_path = os.path.join(save_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=150)
        print(f"ROC curve saved to: {roc_path}")
    else:
        plt.show()
    plt.close()


plot_confusion_matrix(
    all_labels, all_predictions, CLASS_NAMES,
    save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png")
)

plot_pr_roc_curves(
    all_labels, all_probabilities, NUM_CLASSES, CLASS_NAMES,
    save_dir=RESULTS_DIR
)
