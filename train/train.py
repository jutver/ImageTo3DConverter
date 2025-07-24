import os, json, numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2, scipy.ndimage
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==== Dataset ====
class Pix3DDataset(Dataset):
    def __init__(self, json_path, base_dir, voxel_dim=32, transform=None):
        self.transform = transform
        self.samples = []
        self.voxel_dim = voxel_dim
        self.base_path = base_dir
        self.voxel_root = os.path.join(base_dir, "voxel_npy")

        with open(json_path, "r") as f:
            data = json.load(f)

        for item in data:
            img_rel = item["img"]                            # img/chair/0001.png
            img_path = os.path.join(base_dir, img_rel)
            voxel_rel = img_rel.replace("img/", "").replace(".png", ".npy").replace(".jpg", ".npy")
            voxel_path = os.path.join(self.voxel_root, voxel_rel)

            if os.path.exists(img_path) and os.path.exists(voxel_path):
                self.samples.append((img_path, voxel_path))

        print(f"[INFO] Tổng mẫu hợp lệ: {len(self.samples)}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, voxel_path = self.samples[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        img = self.transform(img) if self.transform else torch.tensor(img).permute(2, 0, 1)

        voxel = np.load(voxel_path)
        voxel = scipy.ndimage.zoom(voxel, zoom=(self.voxel_dim/128,)*3, order=0)
        voxel = torch.tensor(voxel, dtype=torch.float32).unsqueeze(0)  # (1,D,H,W)

        return img, voxel

# ==== Model ====
class ResNetVoxel(nn.Module):
    def __init__(self, voxel_dim=32):
        super().__init__()
        self.voxel_dim = voxel_dim
        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Identity()
        self.encoder = resnet
        self.decoder = nn.Sequential(
            nn.Linear(512, 2048), nn.ReLU(),
            nn.Linear(2048, voxel_dim**3), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(-1, 1, self.voxel_dim, self.voxel_dim, self.voxel_dim)

# ==== Training ====
def train(model, loader, optimizer, criterion, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for img, voxel in loader:
            img, voxel = img.to(device), voxel.to(device)
            output = model(img)
            loss = criterion(output, voxel)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(loader):.6f}")

# ==== Evaluation ====
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for img, voxel in loader:
            img, voxel = img.to(device), voxel.to(device)
            output = model(img)
            y_pred.append(output.cpu().numpy().reshape(output.size(0), -1))
            y_true.append(voxel.cpu().numpy().reshape(voxel.size(0), -1))
    y_pred, y_true = np.concatenate(y_pred), np.concatenate(y_true)
    return mean_absolute_error(y_true, y_pred), mean_squared_error(y_true, y_pred), r2_score(y_true, y_pred)

# ==== MAIN ====
if __name__ == "__main__":
    base_path = "ImageTo3DConverter/train/data"
    json_file = os.path.join(base_path, "pix3d.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    dataset = Pix3DDataset(json_file, base_path, voxel_dim=32, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = ResNetVoxel(voxel_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    train(model, dataloader, optimizer, criterion, device, epochs=30)

    save_path = os.path.join(base_path, "resnet_voxel_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n💾 Mô hình đã được lưu tại: {save_path}")

    mae, mse, r2 = evaluate(model, dataloader, device)
    print(f"\n📊 Đánh giá mô hình:")
    print(f"   🔸 MAE: {mae:.6f}")
    print(f"   🔸 MSE: {mse:.6f}")
    print(f"   🔸 R²:  {r2:.6f}")
