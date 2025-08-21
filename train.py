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

# Checkpoint: try to load if exists (continue training); failure won't crash.
checkpoint_path = "best_classifier.pth"
if os.path.exists(checkpoint_path):
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint '{checkpoint_path}'. Will continue training from it.")
    except Exception as e:
        print(f"Warning: failed to load checkpoint '{checkpoint_path}': {e}. Starting fresh.")
else:
    print(f"Checkpoint '{checkpoint_path}' not found. Training from scratch.")

# ---- 分阶段训练配置 ----
# 默认策略：先训练 head（fc）若干 epoch，再解冻 layer4 并微调，再可选全量微调
head_epochs = 2
layer4_epochs = 5
# full_epochs 为剩余 epoch（如果为 0，则不执行全量微调）
full_epochs = max(0, num_epochs - head_epochs - layer4_epochs)

def set_bn_eval(m):
    """Set all BatchNorm layers to eval mode (disable running stats update)."""
    for module in m.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()

def set_trainable(model, train_fc=False, train_layer4=False, train_all=False):
    """Set requires_grad for model parts according to phase."""
    if train_all:
        for p in model.parameters():
            p.requires_grad = True
        return

    # default: freeze all
    for p in model.parameters():
        p.requires_grad = False
    if train_layer4:
        for p in model.layer4.parameters():
            p.requires_grad = True
    if train_fc:
        for p in model.fc.parameters():
            p.requires_grad = True

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_optimizer(model, base_lr, weight_decay=1e-4):
    """Create optimizer with sensible param groups: fc, layer4, others(if trainable)."""
    fc_params = [p for p in model.fc.parameters() if p.requires_grad]
    layer4_params = [p for p in model.layer4.parameters() if p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and ("fc." not in n and "layer4." not in n)]

    param_groups = []
    if fc_params:
        param_groups.append({'params': fc_params, 'lr': base_lr})
    if layer4_params:
        param_groups.append({'params': layer4_params, 'lr': base_lr * 0.1})
    if other_params:
        # smaller lr for other pretrained layers if they are trainable
        param_groups.append({'params': other_params, 'lr': base_lr * 0.01})

    if not param_groups:
        # fallback: no trainable params -> optimizer on all params (safe guard)
        return optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    return optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)

# 初始化：确定 epoch 0 的 phase 并创建 optimizer
def get_phase_for_epoch(ep):
    if ep < head_epochs:
        return 'head'
    if ep < head_epochs + layer4_epochs:
        return 'layer4'
    return 'full'

current_phase = get_phase_for_epoch(0)
if current_phase == 'head':
    set_trainable(model, train_fc=True, train_layer4=False, train_all=False)
    # 对少量可训练参数时，通常将 BN 设为 eval 防止 running stats 被噪声破坏
    set_bn_eval(model)
elif current_phase == 'layer4':
    set_trainable(model, train_fc=True, train_layer4=True, train_all=False)
    set_bn_eval(model)
else:
    set_trainable(model, train_all=True)

print(f"Starting epoch 0 in phase '{current_phase}', trainable params: {count_trainable_params(model)}")
optimizer = make_optimizer(model, learning_rate, weight_decay=weight_decay)

# ---------------------------
# 损失函数
# ---------------------------
criterion = nn.CrossEntropyLoss()

# ---------------------------
# 训练循环 + 验证集评估 + Top-3 + 保存最佳模型
# ---------------------------
best_val_acc = 0.0

for epoch in range(num_epochs):
    # 每个 epoch 开始时检查阶段是否变化；若变化则更新可训练参数并重建 optimizer，保存阶段快照
    new_phase = get_phase_for_epoch(epoch)
    if new_phase != current_phase:
        current_phase = new_phase
        if current_phase == 'head':
            set_trainable(model, train_fc=True, train_layer4=False, train_all=False)
        elif current_phase == 'layer4':
            set_trainable(model, train_fc=True, train_layer4=True, train_all=False)
        else:
            set_trainable(model, train_all=True)

        # 重建 optimizer（按照新的可训练参数分组）
        optimizer = make_optimizer(model, learning_rate, weight_decay=weight_decay)
        print(f"--- Phase changed to '{current_phase}' at epoch {epoch}. Trainable params: {count_trainable_params(model)} ---")
        # 保存阶段开始的模型快照（便于回滚）
        snapshot_path = f"phase_snapshot_{current_phase}_epoch{epoch}.pth"
        torch.save(model.state_dict(), snapshot_path)
        print(f"Saved phase snapshot: {snapshot_path}")

    model.train()
    # 在 head 和 layer4 阶段通常不更新 BN 的 running stats
    if current_phase in ('head', 'layer4'):
        set_bn_eval(model)

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
