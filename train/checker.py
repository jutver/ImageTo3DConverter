import os
import json
import numpy as np
import cv2
import re

# === Cấu hình đường dẫn ===
base_path = "ImageTo3DConverter/train/data"
json_file = os.path.join(base_path, "pix3d.json")

img_dir = os.path.join(base_path, "img")
mask_dir = os.path.join(base_path, "mask")
voxel_dir = os.path.join(base_path, "voxel_npy")
EXPECTED_VOXEL_SHAPE = (128, 128, 128)

# === Danh sách lỗi ===
missing_img = []
missing_mask = []
missing_voxel = []
wrong_img_shape = []
wrong_voxel_shape = []

# === Load JSON ===
with open(json_file, "r") as f:
    data = json.load(f)

# === Duyệt từng mẫu ===
for item in data:
    rel_img_path = item["img"]
    rel_mask_path = item["mask"]

    # Chuyển rel_voxel_path: "img/bed/0001.jpg" → "bed/0001.npy"
    rel_voxel_path = re.sub(r"^img[\\/]", "", rel_img_path)
    rel_voxel_path = os.path.splitext(rel_voxel_path)[0] + ".npy"

    img_path = os.path.join(base_path, rel_img_path)
    mask_path = os.path.join(base_path, rel_mask_path)
    voxel_path = os.path.join(voxel_dir, rel_voxel_path)

    uid = rel_img_path

    # === Kiểm tra ảnh ===
    if not os.path.exists(img_path):
        missing_img.append(uid)
    else:
        img = cv2.imread(img_path)
        if img is None or img.ndim != 3 or img.shape[2] != 3:
            wrong_img_shape.append(uid)

    # === Kiểm tra mask ===
    if not os.path.exists(mask_path):
        missing_mask.append(uid)

    # === Kiểm tra voxel ===
    if not os.path.exists(voxel_path):
        missing_voxel.append(uid)
    else:
        try:
            voxel = np.load(voxel_path)
            if voxel.shape != EXPECTED_VOXEL_SHAPE:
                wrong_voxel_shape.append(uid)
        except Exception as e:
            print(f"[ERROR] Lỗi đọc voxel {uid}: {e}")
            wrong_voxel_shape.append(uid)

# === In kết quả ===
def print_list(title, items):
    print(f"\n❌ {title}: {len(items)} file(s)")
    for i in items[:10]:
        print(f" - {i}")
    if len(items) > 10:
        print(f"... ({len(items) - 10} dòng nữa)")

print_list("Thiếu ảnh", missing_img)
print_list("Thiếu mask", missing_mask)
print_list("Thiếu voxel", missing_voxel)
print_list("Ảnh sai shape", wrong_img_shape)
print_list("Voxel sai shape", wrong_voxel_shape)

# === Tổng kết ===
total = len(data)
errors = len(set(missing_img + missing_mask + missing_voxel + wrong_img_shape + wrong_voxel_shape))
print(f"\n📊 Tổng cộng: {total} mẫu — ✅ OK: {total - errors} — ❌ Lỗi: {errors}")
