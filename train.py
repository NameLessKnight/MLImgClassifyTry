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

# 检查点加载（恢复训练）
def resume_from_checkpoint(path, model, device):
        """
        尝试从 checkpoint 恢复模型并返回 (loaded_epoch, loaded_optimizer_state)。

        支持两种格式：
        1) 完整 checkpoint（字典，包含 'epoch','model','optimizer'），这是推荐格式；
        2) 旧的仅包含 model.state_dict() 的格式（兼容加载）。

        如果加载失败，函数会打印警告并返回 (None, None)。

        参数：
            - path: checkpoint 文件路径
            - model: 要加载权重的模型实例（会就地修改）
            - device: 加载到的设备（cpu 或 cuda）

        返回：
            (loaded_epoch, loaded_optimizer_state) 或 (None, None)
        """
    if not os.path.exists(path):
        print(f"Checkpoint '{path}' not found. Training from scratch.")
        return None, None

    try:
        ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            loaded_optimizer = ckpt.get('optimizer', None)
            loaded_epoch = ckpt.get('epoch', None)
            print(f"Loaded full checkpoint '{path}' (epoch={loaded_epoch}).")
            return loaded_epoch, loaded_optimizer
        else:
            model.load_state_dict(ckpt)
            print(f"Loaded model state_dict from '{path}'.")
            return None, None
    except Exception as e:
        print(f"Warning: failed to load checkpoint '{path}': {e}. Starting fresh.")
        return None, None


checkpoint_path = "best_classifier.pth"
loaded_epoch, loaded_optimizer_state = resume_from_checkpoint(checkpoint_path, model, device)

# ---- 分阶段训练配置 ----
# 默认策略：先训练 head（fc）若干 epoch，再解冻 layer4 并微调，再可选全量微调
head_epochs = 2
layer4_epochs = 5
# full_epochs 为剩余 epoch（如果为 0，则不执行全量微调）
full_epochs = max(0, num_epochs - head_epochs - layer4_epochs)

def set_bn_eval(m):
    """
    将模型中的所有 BatchNorm 层切换到 eval 模式，停止更新 running mean/var。

    为什么要这样做：在微调（只训练部分层）时，如果 batch size 很小，
    更新 BN 的 running stats 会被噪声影响，导致性能不稳定。
    所以在只训练 head 或少量层时，通常将 BN 固定为 eval。
    """
    for module in m.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()

def set_trainable(model, train_fc=False, train_layer4=False, train_all=False):
    """
    根据当前训练阶段设置哪些参数参与训练（即设置 requires_grad）。

    参数说明：
      - train_fc: 是否训练分类头（model.fc）
      - train_layer4: 是否训练最后一组残差块（model.layer4）
      - train_all: 是否训练全部参数

    典型用法：
      - 只训练 head： set_trainable(model, train_fc=True)
      - 解冻 layer4： set_trainable(model, train_fc=True, train_layer4=True)
      - 全量训练： set_trainable(model, train_all=True)
    """
    if train_all:
        for p in model.parameters():
            p.requires_grad = True
        return

    # 默认先冻结所有参数（不参与梯度计算）
    for p in model.parameters():
        p.requires_grad = False
    # 根据标志有选择地解冻某些模块
    if train_layer4:
        for p in model.layer4.parameters():
            p.requires_grad = True
    if train_fc:
        for p in model.fc.parameters():
            p.requires_grad = True

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_optimizer(model, base_lr, weight_decay=1e-4):
    """
    根据当前哪些参数被设置为可训练，创建带有参数组（param groups）的优化器。

    目的：对不同模块使用不同的学习率，例如：
      - 新初始化的分类头（fc）使用 base_lr（较大）以快速学习；
      - 微调的预训练层（layer4）使用较小的 lr（base_lr * 0.1）；
      - 其他可训练的预训练层使用更小 lr（base_lr * 0.01）。

    返回一个 Adam 优化器实例。
    """
    fc_params = [p for p in model.fc.parameters() if p.requires_grad]
    layer4_params = [p for p in model.layer4.parameters() if p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and ("fc." not in n and "layer4." not in n)]

    param_groups = []
    if fc_params:
        param_groups.append({'params': fc_params, 'lr': base_lr})
    if layer4_params:
        param_groups.append({'params': layer4_params, 'lr': base_lr * 0.1})
    if other_params:
        # 如果还有其他可训练的预训练层，给更小的 lr
        param_groups.append({'params': other_params, 'lr': base_lr * 0.01})

    if not param_groups:
        # 兜底：没有可训练参数则将所有参数放入 optimizer（一般不会发生）
        return optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    return optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)

# 初始化：确定 epoch 0 的 phase 并创建 optimizer
def get_phase_for_epoch(ep):
    """
    根据 epoch 返回当前处于哪个训练阶段：'head'、'layer4' 或 'full'。
    这个划分是基于 head_epochs 和 layer4_epochs 的累积。
    """
    if ep < head_epochs:
        return 'head'
    if ep < head_epochs + layer4_epochs:
        return 'layer4'
    return 'full'

# 根据是否加载了 checkpoint 决定从哪个 epoch 开始（start_epoch）
start_epoch = 0
if loaded_epoch is not None:
    start_epoch = loaded_epoch + 1

# 初始化 current_phase 与可训练参数，基于 start_epoch
current_phase = get_phase_for_epoch(start_epoch)
if current_phase == 'head':
    set_trainable(model, train_fc=True, train_layer4=False, train_all=False)
    set_bn_eval(model)
elif current_phase == 'layer4':
    set_trainable(model, train_fc=True, train_layer4=True, train_all=False)
    set_bn_eval(model)
else:
    set_trainable(model, train_all=True)

# 创建 optimizer（基于当前可训练参数）并尝试恢复 optimizer state
optimizer = make_optimizer(model, learning_rate, weight_decay=weight_decay)
if loaded_optimizer_state is not None:
    try:
        optimizer.load_state_dict(loaded_optimizer_state)
        print(f"Restored optimizer state from checkpoint; resuming from epoch {start_epoch}.")
    except Exception as e:
        print(f"Warning: failed to restore optimizer state: {e}. Continuing with fresh optimizer.")
elif loaded_epoch is not None:
    print(f"Resuming from epoch {start_epoch} (no optimizer state in checkpoint).")

print(f"Starting epoch {start_epoch} in phase '{current_phase}', trainable params: {count_trainable_params(model)}")

# ---------------------------
# 损失函数
# ---------------------------
criterion = nn.CrossEntropyLoss()

# ---------------------------
# 训练循环 + 验证集评估 + Top-3 + 保存最佳模型
# ---------------------------
best_val_acc = 0.0

for epoch in range(start_epoch, num_epochs):
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
        # 保存完整字典，包含 model + optimizer(state_dict) + epoch，便于完整恢复训练
        snapshot = {'epoch': epoch, 'model': model.state_dict()}
        try:
            snapshot['optimizer'] = optimizer.state_dict()
        except Exception:
            snapshot['optimizer'] = None
        torch.save(snapshot, snapshot_path)
        print(f"Saved phase snapshot: {snapshot_path} (includes optimizer state if available)")

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

    # 保存最佳模型（完整 checkpoint：包含 epoch、model、optimizer）
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_path = "best_classifier.pth"
        ckpt = {'epoch': epoch, 'model': model.state_dict()}
        try:
            ckpt['optimizer'] = optimizer.state_dict()
        except Exception:
            ckpt['optimizer'] = None
        torch.save(ckpt, best_path)
        print(f"  → 新最佳模型保存 (完整 checkpoint): {best_path}, Val Acc={val_acc:.2f}%")

print("训练完成，最佳模型已保存: best_classifier.pth")
