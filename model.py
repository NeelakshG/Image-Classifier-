import os
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


class CombinedDataset(Dataset):
    """
    Training dataset:
      - In-domain images: class label = 0..(C-1), domain = 1
      - Out-domain images: class label = -1 (ignored), domain = 0
    """

    def __init__(self, in_domain_path, out_domain_path,
                 transform_in, transform_out):

        self.transform_in = transform_in
        self.transform_out = transform_out
        self.samples = []

        in_classes = sorted(os.listdir(in_domain_path))
        self.num_classes = len(in_classes)

        for idx, cname in enumerate(in_classes):
            class_dir = os.path.join(in_domain_path, cname)
            if not os.path.isdir(class_dir):
                continue
            for img in os.listdir(class_dir):
                if img.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, img), idx, 1))

        unl = os.path.join(out_domain_path, "unlabelled")
        if os.path.exists(unl):
            for img in os.listdir(unl):
                if img.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(unl, img), -1, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_label, domain_label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if domain_label == 1:
            img = self.transform_in(img)
        else:
            img = self.transform_out(img)

        return img, class_label, domain_label



class SmallCNN(nn.Module):
    def __init__(self, num_classes=10, input_size=128):
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
        )

        final = input_size // 8  
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * final * final, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def learn(path_to_in_domain_train, path_to_out_domain_train):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_in = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    transform_out = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    tmp = datasets.ImageFolder(path_to_in_domain_train)
    num_classes = len(tmp.classes)

    dataset = CombinedDataset(
        path_to_in_domain_train,
        path_to_out_domain_train,
        transform_in,
        transform_out
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SmallCNN(num_classes=num_classes).to(device)
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    uniform_target = torch.full((1, num_classes), 1.0 / num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 100

    for epoch in range(EPOCHS):
        if epoch < 10:
            KL_W = 0.0
        elif epoch < 20:
            KL_W = 0.5
        else:
            KL_W = 1.0

        model.train()

        for images, labels, domains in loader:
            images = images.to(device)
            labels = labels.to(device)
            domains = domains.to(device)

            optimizer.zero_grad()
            logits = model(images)

           
            mask_in = (domains == 1)
            if mask_in.sum() > 0:
                ce = ce_loss(logits[mask_in], labels[mask_in])
            else:
                ce = torch.tensor(0.0, device=device)

            mask_out = (domains == 0)
            if mask_out.sum() > 0:
                log_probs = nn.LogSoftmax(dim=1)(logits[mask_out])
                uniform = uniform_target.repeat(log_probs.size(0), 1)
                kl = kl_loss(log_probs, uniform)
            else:
                kl = torch.tensor(0.0, device=device)

            loss = ce + KL_W * kl
            loss.backward()
            optimizer.step()

    return model

def compute_accuracy(path_to_eval, model):
    """
    Computes accuracy for BOTH:
      - in-domain-eval (labelled)
      - out-domain-eval (also labelled as per assignment)

    Uses ImageFolder → so folders *must* contain class subfolders.
    """

    device = next(model.parameters()).device

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    eval_dataset = datasets.ImageFolder(path_to_eval, transform=transform)
    loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return 0.0

    return correct / total
