import os
import json
import numpy as np
import scipy.io

# Đường dẫn
base_path = "ImageTo3DConverter/train/data"
json_file = os.path.join(base_path, "pix3d.json")
voxel_npy_dir = os.path.join(base_path, "voxel_npy")

# Load JSON
with open(json_file, "r") as f:
    data = json.load(f)

count = 0
errors = 0

for item in data:
    voxel_mat_path = os.path.join(base_path, item["voxel"])
    img_rel_path = item["img"]  # vd: img/bed/0001.png
    voxel_npy_rel_path = img_rel_path.replace(".png", ".npy")
    voxel_npy_path = os.path.join(voxel_npy_dir, voxel_npy_rel_path)

    os.makedirs(os.path.dirname(voxel_npy_path), exist_ok=True)

    try:
        if not os.path.exists(voxel_npy_path):
            mat = scipy.io.loadmat(voxel_mat_path)
            if "voxel" in mat:
                voxel = mat["voxel"]
            elif "voxels" in mat:
                voxel = mat["voxels"]
            else:
                voxel = list(mat.values())[-1]  # fallback

            np.save(voxel_npy_path, voxel)
            count += 1
    except Exception as e:
        print(f"[❌] Lỗi {voxel_mat_path}: {e}")
        errors += 1

print(f"\n✅ Đã convert: {count} voxel .mat → .npy")
print(f"❌ Lỗi: {errors} file")
