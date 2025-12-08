import os
import zipfile
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from scipy.io import wavfile  # 用于生成/保存噪声

# -------------------------- 配置参数 --------------------------
class Config:
    # 数据集路径配置
    ZIP_PATH = "./speech-emotion-recognition-ravdess-data.360zip"  # 数据集压缩包路径
    DATASET_PATH = "./speech-emotion-recognition-ravdess-data"  # 解压后数据集目录
    SAVE_FEATURE_PATH = "./preprocessed_features"  # 预处理+增强后特征保存目录
    NOISE_SAVE_PATH = "./noise_samples"  # 噪声样本保存目录（若未提供DEMAND库，自动生成）
    
    # 音频预处理参数
    TARGET_SR = 16000  # 目标采样率
    TARGET_DURATION = 4  # 音频统一时长（4秒）
    N_MELS = 128  # Log-Mel谱图Mel频段数
    TIME_STEPS = 64  # Log-Mel谱图时间帧数量
    
    # 情感标签映射
    EMOTION_MAP = {
        1: "neutral", 2: "calm", 3: "happy", 4: "sad",
        5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"
    }
    TARGET_EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "surprised"]  # 目标6类
    
    # 数据增强参数（时间拉伸0.8-1.2随机浮点数）
    AUGMENT_TIMES = 3  # 每个原始样本生成3个增强样本
    TIME_STRETCH_MIN = 0.8  # 时间拉伸最小值
    TIME_STRETCH_MAX = 1.2  # 时间拉伸最大值
    VOLUME_SHIFT_RANGE = (-3, 4)  # 音量扰动范围（±3dB）
    
    # 噪声注入参数
    ADD_NOISE = True  # 是否添加噪声
    NOISE_TYPES = ["white", "pink", "office"]  # 噪声类型（白噪声/粉红噪声/办公室噪声）
    SNR_RANGE = (20, 30)  # 信噪比（20-30dB，数值越大噪声越小）
    NOISE_SAMPLES_NUM = 10  # 生成噪声样本数量（若无外部噪声库）

# -------------------------- 噪声生成/加载函数 --------------------------
def generate_noise_samples(config):
    """生成噪声样本（白噪声/粉红噪声），若无外部噪声库时使用"""
    os.makedirs(config.NOISE_SAVE_PATH, exist_ok=True)
    noise_files = []
    
    for noise_type in config.NOISE_TYPES:
        if noise_type == "office":
            # 模拟办公室噪声（混合白噪声+低频粉红噪声）
            for i in range(config.NOISE_SAMPLES_NUM):
                noise_len = config.TARGET_SR * config.TARGET_DURATION
                # 白噪声
                white_noise = np.random.normal(0, 0.005, noise_len)
                # 粉红噪声（1/f频谱）
                pink_noise = np.random.normal(0, 0.003, noise_len)
                pink_noise = librosa.effects.preemphasis(pink_noise, coef=0.97)
                # 混合
                office_noise = 0.7 * white_noise + 0.3 * pink_noise
                # 保存噪声
                noise_path = os.path.join(config.NOISE_SAVE_PATH, f"{noise_type}_noise_{i}.wav")
                wavfile.write(noise_path, config.TARGET_SR, (office_noise * 32767).astype(np.int16))
                noise_files.append(noise_path)
        else:
            # 生成白噪声/粉红噪声
            for i in range(config.NOISE_SAMPLES_NUM):
                noise_len = config.TARGET_SR * config.TARGET_DURATION
                if noise_type == "white":
                    noise = np.random.normal(0, 0.005, noise_len)
                elif noise_type == "pink":
                    noise = np.random.normal(0, 0.003, noise_len)
                    noise = librosa.effects.preemphasis(noise, coef=0.97)
                # 保存噪声
                noise_path = os.path.join(config.NOISE_SAVE_PATH, f"{noise_type}_noise_{i}.wav")
                wavfile.write(noise_path, config.TARGET_SR, (noise * 32767).astype(np.int16))
                noise_files.append(noise_path)
    return noise_files

def load_noise_samples(config):
    """加载噪声样本（优先外部库，无则生成）"""
    # 若有外部噪声库（如DEMAND），替换为你的噪声库路径
    external_noise_path = "./DEMAND"
    if os.path.exists(external_noise_path):
        noise_files = []
        for root, _, files in os.walk(external_noise_path):
            for file in files:
                if file.endswith(".wav"):
                    noise_files.append(os.path.join(root, file))
        print(f"加载外部噪声库，共{len(noise_files)}条噪声样本")
    else:
        print("未找到外部噪声库，自动生成噪声样本...")
        noise_files = generate_noise_samples(config)
        print(f"生成{len(noise_files)}条噪声样本，保存至{config.NOISE_SAVE_PATH}")
    return noise_files

def add_noise_to_audio(audio, sr, noise_files, config):
    """为音频注入噪声（随机选择噪声类型+信噪比）"""
    # 随机选一个噪声文件
    noise_path = random.choice(noise_files)
    noise, _ = librosa.load(noise_path, sr=sr)
    
    # 匹配噪声长度与音频长度
    if len(noise) > len(audio):
        noise = noise[:len(audio)]
    else:
        noise = np.pad(noise, (0, len(audio) - len(noise)), mode='constant')
    
    # 计算信噪比，调整噪声强度
    snr = random.uniform(*config.SNR_RANGE)
    audio_power = np.sum(audio ** 2) / len(audio)
    noise_power = np.sum(noise ** 2) / len(noise)
    noise_scale = np.sqrt(audio_power / (10 ** (snr / 10) * noise_power))
    
    # 注入噪声
    audio_noisy = audio + noise_scale * noise
    # 防止音频幅值溢出
    audio_noisy = np.clip(audio_noisy, -1.0, 1.0)
    return audio_noisy

# -------------------------- 数据增强函数 --------------------------
def augment_audio(audio, sr, config):
    """音频增强：时间拉伸（0.8-1.2随机浮点数）+音量扰动"""
    # 生成0.8-1.2之间的随机浮点数
    stretch_ratio = random.uniform(config.TIME_STRETCH_MIN, config.TIME_STRETCH_MAX)
    
    # 时间拉伸
    audio_stretched = librosa.effects.time_stretch(audio, rate=stretch_ratio)
    
    # 统一长度（拉伸后长度变化，需重新对齐）
    target_len = sr * config.TARGET_DURATION
    if len(audio_stretched) < target_len:
        audio_stretched = np.pad(audio_stretched, (0, target_len - len(audio_stretched)), mode='constant')
    else:
        audio_stretched = audio_stretched[:target_len]
    
    # 音量扰动
    db_shift = random.randint(*config.VOLUME_SHIFT_RANGE)
    audio_vol = librosa.amplitude_to_db(audio_stretched) + db_shift
    audio_vol = librosa.db_to_amplitude(audio_vol)
    
    return audio_vol

# -------------------------- 基础预处理函数（移除降噪步骤） --------------------------
def unzip_dataset(config):
    """解压数据集（若未解压）"""
    if not os.path.exists(config.DATASET_PATH):
        os.makedirs(config.DATASET_PATH, exist_ok=True)
        print(f"正在解压数据集到 {config.DATASET_PATH}...")
        with zipfile.ZipFile(config.ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(config.DATASET_PATH)
        print("数据集解压完成！")
    else:
        print(f"数据集已解压，跳过解压步骤")

def parse_ravdess_filename(file):
    """解析RAVDESS文件名，提取情感标签"""
    parts = file.split("-")
    emotion_code = int(parts[2])
    return Config.EMOTION_MAP.get(emotion_code, "unknown")

def preprocess_base_audio(audio_path, config):
    """基础预处理：加载+采样率转换+统一时长（移除降噪，避免版本兼容问题）"""
    # 加载音频
    y, sr = librosa.load(audio_path, sr=config.TARGET_SR)
    # 统一时长
    target_len = config.TARGET_SR * config.TARGET_DURATION
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]
    return y, sr

def extract_log_mel(audio, sr, config):
    """提取Log-Mel谱图特征"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=config.N_MELS, fmax=sr//2
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    # 统一时间帧
    if log_mel.shape[1] < config.TIME_STEPS:
        log_mel = np.pad(log_mel, ((0, 0), (0, config.TIME_STEPS - log_mel.shape[1])), mode='constant')
    else:
        log_mel = log_mel[:, :config.TIME_STEPS]
    # 增加通道维度
    log_mel = log_mel[..., np.newaxis]
    return log_mel

# -------------------------- 完整预处理+增强流程 --------------------------
def full_preprocess(config):
    # 1. 解压数据集
    unzip_dataset(config)
    
    # 2. 加载噪声样本（用于注入噪声）
    if config.ADD_NOISE:
        noise_files = load_noise_samples(config)
    
    # 3. 创建特征保存目录
    os.makedirs(config.SAVE_FEATURE_PATH, exist_ok=True)
    
    # 4. 遍历处理所有音频
    all_features = []
    all_labels = []
    all_audio_paths = []
    all_aug_types = []  # 记录样本类型：原始/增强/噪声
    
    for root, dirs, files in os.walk(config.DATASET_PATH):
        for file in tqdm(files, desc="处理音频（含增强+噪声）"):
            if not file.endswith(".wav"):
                continue
            
            # 解析情感标签，仅保留目标6类
            emotion = parse_ravdess_filename(file)
            if emotion not in config.TARGET_EMOTIONS:
                continue
            
            audio_path = os.path.join(root, file)
            # 基础预处理（加载+统一时长）
            audio_base, sr = preprocess_base_audio(audio_path, config)
            # 提取原始特征
            log_mel_base = extract_log_mel(audio_base, sr, config)
            all_features.append(log_mel_base)
            all_labels.append(emotion)
            all_audio_paths.append(audio_path)
            all_aug_types.append("original")
            
            # 生成增强样本（时间拉伸+音量扰动）
            for aug_idx in range(config.AUGMENT_TIMES):
                audio_aug = augment_audio(audio_base, sr, config)
                log_mel_aug = extract_log_mel(audio_aug, sr, config)
                all_features.append(log_mel_aug)
                all_labels.append(emotion)
                all_audio_paths.append(audio_path)
                all_aug_types.append(f"aug_{aug_idx+1}")
            
            # 生成噪声样本（注入噪声）
            if config.ADD_NOISE:
                audio_noisy = add_noise_to_audio(audio_base, sr, noise_files, config)
                log_mel_noisy = extract_log_mel(audio_noisy, sr, config)
                all_features.append(log_mel_noisy)
                all_labels.append(emotion)
                all_audio_paths.append(audio_path)
                all_aug_types.append("noisy")
    
    # 5. 转换为数组，保存结果
    features_np = np.array(all_features, dtype=np.float32)
    # 保存特征（npy格式，训练时直接加载）
    np.save(os.path.join(config.SAVE_FEATURE_PATH, "log_mel_features_aug.npy"), features_np)
    # 保存标签/路径/样本类型（CSV格式）
    df = pd.DataFrame({
        "audio_path": all_audio_paths,
        "emotion": all_labels,
        "aug_type": all_aug_types
    })
    df.to_csv(os.path.join(config.SAVE_FEATURE_PATH, "labels_aug.csv"), index=False)
    
    # 打印处理结果
    print("\n==================== 预处理完成 ====================")
    print(f"原始样本数：{len([t for t in all_aug_types if t == 'original'])}")
    print(f"增强样本数：{len([t for t in all_aug_types if 'aug' in t])}")
    print(f"噪声样本数：{len([t for t in all_aug_types if t == 'noisy'])}")
    print(f"总样本数：{len(features_np)}")
    print(f"特征形状：{features_np.shape}（样本数 × Mel频段 × 时间帧 × 通道）")
    print(f"特征保存路径：{os.path.join(config.SAVE_FEATURE_PATH, 'log_mel_features_aug.npy')}")
    print(f"标签保存路径：{os.path.join(config.SAVE_FEATURE_PATH, 'labels_aug.csv')}")

# -------------------------- 训练时调用数据集的示例 --------------------------
def load_preprocessed_data(config):
    """训练时直接调用预处理后的数据集"""
    # 加载特征
    features = np.load(os.path.join(config.SAVE_FEATURE_PATH, "log_mel_features_aug.npy"))
    # 加载标签
    df = pd.read_csv(os.path.join(config.SAVE_FEATURE_PATH, "labels_aug.csv"))
    labels = df["emotion"].values
    
    # 标签编码（转换为数字，适配模型训练）
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    
    # 保存标签编码器（推理时使用）
    import pickle
    with open(os.path.join(config.SAVE_FEATURE_PATH, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    
    print(f"加载数据集完成！")
    print(f"特征形状：{features.shape}")
    print(f"标签类别：{le.classes_}")
    return features, labels_encoded, le

# -------------------------- 执行入口 --------------------------
if __name__ == "__main__":
    # 安装依赖（首次运行可取消注释）
    # os.system("pip install librosa numpy pandas tqdm scipy scikit-learn")
    
    # 执行完整预处理
    full_preprocess(Config)
    
    # 训练时调用数据集（注释掉，需训练时取消注释）
    # features, labels_encoded, le = load_preprocessed_data(Config)