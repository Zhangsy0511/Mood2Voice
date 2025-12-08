import os
import glob
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import torchaudio
import torchaudio.transforms as T
# -----------配置---------

# -------- CBAM Block ----------
class CBAMLayer(nn.Module):
    def __init__(self,channel,reduction=16,kernel_size=7):
        super().__init__()
        # Channel Attention
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.max_pool=nn.AdaptiveMaxPool2d(1)
        self.fc=nn.Sequential(
            nn.Linear(channel,channel//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(channel//reduction,channel,bias=False)
        )
        self.sigmoid_channel=nn.Sigmoid()
        # Spatial Attention
        self.conv_spatial=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2,bias=False)
        self.sigmoid_spatial=nn.Sigmoid()
    def forward(self,x):
        b,c,h,w=x.size()
        # Channel Attention
        avg_out=self.fc(self.avg_pool(x).view(b,c)).view(b,c,1,1)
        max_out=self.fc(self.max_pool(x).view(b,c)).view(b,c,1,1)
        channel_attn=self.sigmoid_channel(avg_out+max_out)
        x=x*channel_attn
        # Spatial Attention
        avg_out=torch.mean(x,dim=1,keepdim=True)
        max_out,_=torch.max(x,dim=1,keepdim=True)
        spatial_attn=self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out,max_out],dim=1)))
        x=x*spatial_attn
        return x
DATA_DIR = r'Mood2Voice\speech-emotion-recognition-ravdess-data'
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
FIXED_T = 300        # 时间帧统一长度
BATCH_SIZE = 16
EPOCHS = 1
LR = 1e-3
WEIGHT_DECAY = 1e-4


EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# -------- CNN Block ----------
class CNNBlock(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.conv=nn.Conv2d(in_c,out_c,kernel_size=3,padding=1)
        self.bn=nn.BatchNorm2d(out_c)
        self.relu=nn.ReLU()
        self.pool=nn.MaxPool2d(kernel_size=(2,2))
    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))

# -------- SE Block ----------
class SELayer(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Linear(channel,channel//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(channel//reduction,channel,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,_,_=x.size()
        y=self.avg_pool(x).view(b,c)
        y=self.fc(y).view(b,c,1,1)
        return x*y
    

class EmotionNet(nn.Module):
    def __init__(self, n_classes=8):
        super(EmotionNet, self).__init__()
        self.block1 = nn.Sequential(
            CNNBlock(1, 32),
            SELayer(32)
        )
        self.block2 = nn.Sequential(
            CNNBlock(32, 64),
            SELayer(64)
        )
        self.block3 = nn.Sequential(
            CNNBlock(64, 128),
            SELayer(128)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )
#pip install soundfile -i https://pypi.tuna.tsinghua.edu.cn/simple
#pip install sox -i https://pypi.tuna.tsinghua.edu.cn/simple
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).flatten(1)
        x = self.classifier(x)
        return x

_mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)
   
def wav_to_mel(filepath: str) -> torch.Tensor:
    """
    使用 torchaudio 读取 wav 并转为统一大小的 Mel 频谱:
    返回 shape: (1, N_MELS, FIXED_T)
    """
    # waveform: (channels, time)
    waveform, sr = torchaudio.load(filepath)
    # 如果采样率不一致，重采样到 SAMPLE_RATE
    if sr != SAMPLE_RATE:
        resample = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resample(waveform)
    # 只取第一通道
    if waveform.size(0) > 1:
        waveform = waveform[0:1, :]
    mel = _mel_transform(waveform)  # (1, n_mels, time)
    # 归一化到 0~1
    mel_db = torchaudio.functional.amplitude_to_DB(
        mel, multiplier=10.0, amin=1e-10, db_multiplier=0.0
    )  # (1, n_mels, T)
    # 简单 min-max 归一化
    mel_min = mel_db.amin(dim=(1, 2), keepdim=True)
    mel_max = mel_db.amax(dim=(1, 2), keepdim=True)
    mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-9)
    # pad/crop 到固定时间步 FIXED_T
    _, _, t = mel_norm.shape
    if t < FIXED_T:
        pad_right = FIXED_T - t
        mel_norm = torch.nn.functional.pad(
            mel_norm, (0, pad_right), mode="constant", value=0.0
        )
    else:
        mel_norm = mel_norm[:, :, :FIXED_T]
    return mel_norm  # (1, n_mels, FIXED_T)

class SERDataset(Dataset):
    def __init__(self, file_paths, label_ids):
        self.file_paths = file_paths
        self.label_ids = label_ids

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        mel = wav_to_mel(path)  # (1, N_MELS, FIXED_T)
        label_id = self.label_ids[idx]
        return mel, torch.tensor(label_id, dtype=torch.long)

def collect_ravdess_files(observed_emotions: dict):
    """
    observed_emotions: {1:'calm', 2:'happy', ...} 你想训练的子集
    从 DATA_DIR 下递归收集 RAVDESS 文件。
    返回: file_paths, labels, label2id
    """
    # 只保留你关心的情绪集合
    target_emos = sorted(list(observed_emotions.values()))
    label2id = {e: i for i, e in enumerate(target_emos)}

    file_paths = []
    labels = []

    pattern = os.path.join(DATA_DIR, "**", "*.wav")
    for file in glob.glob(pattern, recursive=True):
        fname = os.path.basename(file)
        parts = fname.split("-")
        if len(parts) < 3:
            continue
        emo_code = parts[2]  # '05'
        emo_str = EMOTIONS.get(emo_code, None)
        if emo_str is None:
            continue
        if emo_str not in label2id:
            continue
        file_paths.append(file)
        labels.append(label2id[emo_str])

    return file_paths, labels, label2id

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for mel, label in loader:
        mel = mel.to(device)      # (B, 1, N_MELS, FIXED_T)
        label = label.to(device)

        optimizer.zero_grad()
        logits = model(mel)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * mel.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == label).sum().item()
        total_num += label.size(0)

    avg_loss = total_loss / total_num
    acc = total_correct / total_num
    return avg_loss, acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    with torch.no_grad():
        for mel, label in loader:
            mel = mel.to(device)
            label = label.to(device)

            logits = model(mel)
            loss = criterion(logits, label)

            total_loss += loss.item() * mel.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == label).sum().item()
            total_num += label.size(0)

    avg_loss = total_loss / total_num
    acc = total_correct / total_num
    return avg_loss, acc
def main():
    observed_emotions={
        1:'calm',
        2:'happy',
        3:'fearful',
        4:'disgust'
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    file_paths, labels, label2id = collect_ravdess_files(observed_emotions)
    print(f"Label2id: {label2id}")
    if len(file_paths) == 0:
        print("未在 DATA_DIR 下找到 wav 文件，请检查 DATA_DIR 设置。")
        return

    # 2. 训练 / 验证划分
    x_train, x_val, y_train, y_val = train_test_split(
        file_paths,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    train_dataset = SERDataset(x_train, y_train)
    val_dataset = SERDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    # 3. 模型、优化器、损失
    model = EmotionNet(n_classes=len(label2id)).to(device)
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0.0
    best_path = os.path.join(os.path.dirname(__file__), "best_model.pth")

    # 4. 训练循环
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  >> Saved best model: {best_path}, val_acc={best_val_acc:.4f}")

    print("Training finished. Best Val Acc:", best_val_acc)


if __name__ == "__main__":
    main()
    