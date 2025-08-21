import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import shutil

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
def predict_and_sort(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for cls in classes:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

    for fname in os.listdir(input_dir):
        fpath = os.path.join(input_dir, fname)
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        img = Image.open(fpath).convert("RGB")
        img_t = val_transforms(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_t)
            probs = F.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, 1).item()
            pred_cls = classes[pred_idx]

        dst = os.path.join(output_dir, pred_cls, fname)
        shutil.copy(fpath, dst)
        print(f"{fname} → {pred_cls}")

# ---------------------------
# 使用示例
# ---------------------------
if __name__ == "__main__":
    # 单张图片预测
    #test_results = predict_top3("unsorted_images/example.jpg")
    #print("Top-3预测结果：")
    #for cls, prob in test_results:
    #    print(f"{cls}: {prob:.2%}")

    # 批量文件夹分类
    predict_and_sort("unsorted_images", "sorted_images")
    print("批量分类完成，结果在 sorted_images/ 下")
