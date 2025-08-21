import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F

# ---------------------------
# 配置
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "dataset"
batch_size = 32
num_epochs = 15
learning_rate = 1e-4
weight_decay = 1e-4
log_interval = 10  # 每多少 batch 打印一次训练日志

# ---------------------------
# 数据加载与增强
# ---------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transforms)
val_dataset   = datasets.ImageFolder(root=f"{data_dir}/val", transform=val_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

classes = train_dataset.classes

# ---------------------------
# 模型加载 & 防过拟合
# ---------------------------
model = models.resnet18(weights="IMAGENET1K_V1")
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, len(classes))
)
model = model.to(device)

# 冻结卷积层，只微调 layer4 和 fc
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

# ---------------------------
# 损失函数 & 优化器
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                       lr=learning_rate, weight_decay=weight_decay)

# ---------------------------
# 训练循环 + 验证集评估 + Top-3 + 保存最佳模型
# ---------------------------
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 每 log_interval 打印一次训练 batch 信息
        if (batch_idx + 1) % log_interval == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}][Batch {batch_idx+1}/{len(train_loader)}] "
                  f"Loss={running_loss/(batch_idx+1):.4f}, Train Acc={100*correct/total:.2f}%")

    train_acc = 100 * correct / total

    # ---------------------------
    # 验证集评估
    # ---------------------------
    model.eval()
    val_correct, val_total = 0, 0
    top3_correct, top3_total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Top-1
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            # Top-3
            top3_prob, top3_idx = torch.topk(F.softmax(outputs, dim=1), 3, dim=1)
            for i in range(labels.size(0)):
                if labels[i] in top3_idx[i]:
                    top3_correct += 1
                top3_total += 1

    val_acc = 100 * val_correct / val_total
    val_top3_acc = 100 * top3_correct / top3_total

    print(f"Epoch {epoch+1} Summary: "
          f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, Val Top-3 Acc={val_top3_acc:.2f}%, "
          f"Loss={running_loss/len(train_loader):.4f}")

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_classifier.pth")
        print(f"  → 新最佳模型保存, Val Acc={val_acc:.2f}%")

print("训练完成，最佳模型已保存: best_classifier.pth")
