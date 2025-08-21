import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import shutil
import argparse

# ---------------------------
# 配置
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------------------
# 模型加载
# ---------------------------
classes = ['anime', 'camera', 'other']  # 按训练时类别填
model = models.resnet18(weights=None)
num_features = model.fc.in_features
# Match training head: Dropout followed by Linear (keys were saved as fc.1.*)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(num_features, len(classes))
)
model.load_state_dict(torch.load("best_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

# ---------------------------
# Top-3 预测函数
# ---------------------------
def predict_top3(image_path):
    img = Image.open(image_path).convert("RGB")
    img = val_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)
        top3_prob, top3_idx = torch.topk(probs, 3)
    results = [(classes[idx.item()], float(prob.item())) for prob, idx in zip(top3_prob[0], top3_idx[0])]
    return results

# ---------------------------
# 批量文件夹分类
# ---------------------------
def predict_and_sort(input_dir, output_base=None, output_dirs=None, classes_list=None):
    """Predict and move files from input_dir into class folders.

    Args:
        input_dir (str): directory with images to sort.
        output_base (str|None): base directory under which class subfolders will be created.
        output_dirs (dict|None): optional dict mapping class name -> absolute/relative output dir.
        classes_list (list|None): list of class names in order matching model outputs. If None, uses global `classes`.
    """
    if classes_list is None:
        classes_list = classes

    # Build target directories
    targets = {}
    if output_dirs:
        # Use provided per-class dirs (must cover classes in classes_list)
        for cls in classes_list:
            path = output_dirs.get(cls)
            if path is None:
                # fallback to output_base if available, else create under 'sorted_images'
                base = output_base or 'sorted_images'
                path = os.path.join(base, cls)
            targets[cls] = path
    else:
        base = output_base or 'sorted_images'
        for cls in classes_list:
            targets[cls] = os.path.join(base, cls)

    # Create directories
    for path in set(targets.values()):
        os.makedirs(path, exist_ok=True)

    for fname in os.listdir(input_dir):
        fpath = os.path.join(input_dir, fname)
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        try:
            img = Image.open(fpath).convert("RGB")
        except Exception:
            # skip files that PIL cannot open
            print(f"跳过无法打开的文件: {fname}")
            continue

        img_t = val_transforms(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_t)
            probs = F.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, 1).item()
            pred_cls = classes_list[pred_idx]

        dst_dir = targets.get(pred_cls)
        if dst_dir is None:
            # This can occur if the model predicts a class not present in classes_list (e.g., class label mismatch or model/class list inconsistency).
            # Fallback to the first class directory to ensure the file is not lost; review your class list and model if this happens.
            dst_dir = targets.get(classes_list[0])

        dst = os.path.join(dst_dir, fname)

        # If destination exists, add a numeric suffix to avoid overwrite
        if os.path.exists(dst):
            name, ext = os.path.splitext(fname)
            i = 1
            while True:
                new_name = f"{name}_{i}{ext}"
                new_dst = os.path.join(dst_dir, new_name)
                if not os.path.exists(new_dst):
                    dst = new_dst
                    break
                i += 1

        shutil.move(fpath, dst)
        print(f"{fname} → {pred_cls} (moved -> {dst})")

# ---------------------------
# 使用示例
# ---------------------------
if __name__ == "__main__":
    # 单张图片预测
    #test_results = predict_top3("unsorted_images/example.jpg")
    #print("Top-3预测结果：")
    #for cls, prob in test_results:
    #    print(f"{cls}: {prob:.2%}")

    # 批量文件夹分类，可通过参数指定输入目录、输出基目录或每个类别的目标目录
    parser = argparse.ArgumentParser(description="Predict and move images into class folders")
    parser.add_argument('-i', '--input', default='unsorted_images', help='input directory containing images')
    parser.add_argument('-b', '--output-base', default='sorted_images', help='base output directory (will create subfolders for each class)')
    parser.add_argument('--anime-dir', help='output directory for anime class')
    parser.add_argument('--camera-dir', help='output directory for camera class')
    parser.add_argument('--other-dir', help='output directory for other class')
    args = parser.parse_args()

    per_class = None
    if args.anime_dir or args.camera_dir or args.other_dir:
        per_class = {}
        if args.anime_dir:
            per_class['anime'] = args.anime_dir
        if args.camera_dir:
            per_class['camera'] = args.camera_dir
        if args.other_dir:
            per_class['other'] = args.other_dir

    predict_and_sort(args.input, output_base=args.output_base, output_dirs=per_class)
    print(f"批量分类完成，结果在 {args.output_base}/ (或你指定的每类目录) 下")
