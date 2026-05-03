import torch
import torch.nn as nn
import numpy as np
import os
import cv2
from glob import glob
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# =====================================================
# SETTINGS
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_ROOT = r"C:\Users\Nihar Prabhu\CASIA_preprocessed_128\content\drive\MyDrive\CASIA_preprocessed_128"
MODEL_PATH = r"C:\Users\Nihar Prabhu\Trained_models\student_kd_conv.pth"

NUM_CLASSES = 1000
BATCH_SIZE = 64
N_SPLITS = 5   # 5-Fold Cross Validation

# =====================================================
# TRANSFORMS
# =====================================================
transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# =====================================================
# DATASET
# =====================================================
class CASIADataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform

        subjects = sorted(os.listdir(root))
        self.label_map = {sub: i for i, sub in enumerate(subjects)}

        for sub in subjects:
            sub_path = os.path.join(root, sub)

            for eye in ["L", "R"]:
                eye_path = os.path.join(sub_path, eye)

                images = []
                images += glob(os.path.join(eye_path, "*.jpg"))
                images += glob(os.path.join(eye_path, "*.jpeg"))
                images += glob(os.path.join(eye_path, "*.png"))

                for img in images:
                    self.samples.append((img, self.label_map[sub]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (224, 224))

        if self.transform:
            img = self.transform(img)

        return img, label

# =====================================================
# MODEL
# =====================================================
class StudentCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, return_embedding=False):
        x = self.features(x)

        x_flat = torch.flatten(x, 1)
        emb = self.classifier[1](x_flat)
        emb = self.classifier[2](emb)

        if return_embedding:
            return emb

        return self.classifier(x)

# =====================================================
# LOAD MODEL
# =====================================================
model = StudentCNN(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =====================================================
# LOAD DATA
# =====================================================
dataset = CASIADataset(DATASET_ROOT, transform)
labels = np.array([label for _, label in dataset.samples])

# =====================================================
# CROSS VALIDATION
# =====================================================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

fold_auc = []
fold_eer = []

for fold, (_, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):

    print(f"\n===== FOLD {fold+1} =====")

    subset = Subset(dataset, test_idx)
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)

    embeddings = []
    y_true = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            emb = model(x, return_embedding=True)
            emb = torch.nn.functional.normalize(emb, dim=1)

            embeddings.append(emb.cpu().numpy())
            y_true.extend(y)

    embeddings = np.vstack(embeddings)
    y_true = np.array(y_true)

    # Pairwise comparison
    scores = []
    same_person = []

    N = len(embeddings)
    for i in range(N):
        for j in range(i+1, min(i+30, N)):
            sim = np.dot(embeddings[i], embeddings[j])
            scores.append(sim)
            same_person.append(y_true[i] == y_true[j])

    scores = np.array(scores)
    same_person = np.array(same_person)

    # ROC
    fpr, tpr, _ = roc_curve(same_person, scores)
    roc_auc = auc(fpr, tpr)

    # EER
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    fold_auc.append(roc_auc)
    fold_eer.append(eer)

    print(f"AUC: {roc_auc:.4f}")
    print(f"EER: {eer:.4f}")

# =====================================================
# FINAL RESULTS
# =====================================================
print("\n===== CROSS VALIDATION RESULTS =====")
print(f"Mean AUC: {np.mean(fold_auc):.4f} ± {np.std(fold_auc):.4f}")
print(f"Mean EER: {np.mean(fold_eer):.4f} ± {np.std(fold_eer):.4f}")
