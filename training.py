import os
import cv2
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T

# ============================================================
# PATHS
# ============================================================
DATASET_ROOT = r"C:\Users\Nihar Prabhu\CASIA_preprocessed_128\content\drive\MyDrive\CASIA_preprocessed_128"
SAVE_DIR = r"C:\Users\NIHAR PRABHU\Trained_models"

os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
TEACHER_EPOCHS = 8
KD_EPOCHS = 20
LR = 1e-4

# ============================================================
# DATASET
# ============================================================
class CASIADataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform

        subjects = sorted(os.listdir(root))
        self.label_map = {sub: i for i, sub in enumerate(subjects)}

        for sub in subjects:
            sub_path = os.path.join(root, sub)
            if not os.path.isdir(sub_path):
                continue

            for eye in ["L", "R"]:
             eye_path = os.path.join(sub_path, eye)
             if not os.path.isdir(eye_path):
              continue

             images = []
             images += glob(os.path.join(eye_path, "*.jpg"))
             images += glob(os.path.join(eye_path, "*.jpeg"))
             images += glob(os.path.join(eye_path, "*.png"))
             images += glob(os.path.join(eye_path, "*.JPG"))
             images += glob(os.path.join(eye_path, "*.JPEG"))

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

        return img, torch.tensor(label).long()

# ============================================================
# TRANSFORMS
# ============================================================
transform = T.Compose([
    T.ToPILImage(),
    T.RandomRotation(5),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

dataset = CASIADataset(DATASET_ROOT, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

NUM_CLASSES = len(dataset.label_map)
print("Samples:", len(dataset))
print("Classes:", NUM_CLASSES)

# ============================================================
# TEACHER MODEL
# ============================================================
def get_teacher():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(2048, NUM_CLASSES)
    return model

# ============================================================
# STUDENT MODEL (Conv + MaxPool)
# ============================================================
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

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============================================================
# INIT MODELS
# ============================================================
teacher = get_teacher().to(DEVICE)
student = StudentCNN(NUM_CLASSES).to(DEVICE)

# ============================================================
# STAGE 0 — TRAIN TEACHER ⭐
# ============================================================
print("\n=== STAGE 0: TRAINING TEACHER ===")

optimizer_t = optim.Adam(teacher.parameters(), lr=LR)
ce_loss = nn.CrossEntropyLoss()

for epoch in range(TEACHER_EPOCHS):
    teacher.train()
    correct = 0
    total = 0

    for x, y in tqdm(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        out = teacher(x)
        loss = ce_loss(out, y)

        optimizer_t.zero_grad()
        loss.backward()
        optimizer_t.step()

        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    print(f"Teacher Epoch {epoch+1} Acc: {100*correct/total:.2f}%")

torch.save(teacher.state_dict(),
           os.path.join(SAVE_DIR, "teacher_trained_resnet50.pth"))

# ============================================================
# STAGE 1 — KNOWLEDGE DISTILLATION
# ============================================================
print("\n=== STAGE 1: KD TRAINING ===")

def kd_loss(s_out, t_out, labels, T=3.0, alpha=0.7):
    kd = nn.KLDivLoss(reduction="batchmean")(
        nn.LogSoftmax(dim=1)(s_out/T),
        nn.Softmax(dim=1)(t_out/T)
    ) * (T*T)

    ce = nn.CrossEntropyLoss()(s_out, labels)
    return alpha*kd + (1-alpha)*ce

optimizer_s = optim.Adam(student.parameters(), lr=LR)

for epoch in range(KD_EPOCHS):
    student.train()
    teacher.eval()

    correct = 0
    total = 0

    for x, y in tqdm(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.no_grad():
            t_out = teacher(x)

        s_out = student(x)
        loss = kd_loss(s_out, t_out, y)

        optimizer_s.zero_grad()
        loss.backward()
        optimizer_s.step()

        pred = s_out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    print(f"KD Epoch {epoch+1} Acc: {100*correct/total:.2f}%")

torch.save(student.state_dict(),
           os.path.join(SAVE_DIR, "student_kd_conv.pth"))

# ============================================================
# STAGE 2 — PRUNING
# ============================================================
print("\n=== STAGE 2: PRUNING ===")

import torch.nn.utils.prune as prune

for module in student.modules():
    if isinstance(module, nn.Conv2d):
        prune.ln_structured(module, name="weight", amount=0.3, n=2, dim=0)

torch.save(student.state_dict(),
           os.path.join(SAVE_DIR, "student_pruned_conv.pth"))

# Remove pruning mask
for module in student.modules():
    if isinstance(module, nn.Conv2d):
        try:
            prune.remove(module, "weight")
        except:
            pass

# ============================================================
# STAGE 3 — QUANTIZATION
# ============================================================
print("\n=== STAGE 3: QUANTIZATION ===")

student_cpu = student.to("cpu")

qstudent = torch.quantization.quantize_dynamic(
    student_cpu,
    {nn.Linear},
    dtype=torch.qint8
)

torch.save(qstudent.state_dict(),
           os.path.join(SAVE_DIR, "student_quantized_conv.pth"))

print("\n✅ ALL TRAINING + COMPRESSION DONE")
print("Saved in:", SAVE_DIR)
