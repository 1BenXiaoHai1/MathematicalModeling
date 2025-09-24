import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft
import pywt
from argparse import ArgumentParser, Namespace
import sys
import os
import glob
import re
from pathlib import Path
import math

BP_LOW, BP_HIGH = 500.0, 10000.0 # 带通滤波带宽（Hz）
FILTER_ORDER = 4  # 滤波器阶数
fs_in = 12000  # 输入信号采样频率（Hz）
DEFAULT_FS_GUESS = 12000  # 默认采样频率猜测值
TARGET_FS = 32000  # 重采样目标采样率

# ---- 轴承几何（英寸） ----
GEOM = {
    "DE": {"Nd": 9, "d": 0.3126, "D": 1.537},  # SKF6205
    "FE": {"Nd": 9, "d": 0.2656, "D": 1.122},  # SKF6203
}

def load_data_from_excel(file_path):
    """
    从Excel文件读取数据，动态识别列名
    """
    try:
        df = pd.read_excel(file_path)
        
        # 动态识别列名
        de_col = None
        fe_col = None
        ba_col = None
        rpm_col = None
        
        # 使用正则表达式匹配列名
        for col in df.columns:
            if re.search(r'_DE_time$', col, re.IGNORECASE):
                de_col = col
            elif re.search(r'_FE_time$', col, re.IGNORECASE):
                fe_col = col
            elif re.search(r'_BA_time$', col, re.IGNORECASE):
                ba_col = col
            elif re.search(r'RPM$', col, re.IGNORECASE):
                rpm_col = col
        
        # 检查是否找到所有必要的列，如果没找到则使用默认值
        if not de_col:
            print(f"Warning: DE column not found in {file_path}, using default zeros")
            de_data = np.zeros(len(df))
        else:
            de_data = df[de_col].dropna().values
            if len(de_data) == 0:
                de_data = np.zeros(len(df))
        
        if not fe_col:
            print(f"Warning: FE column not found in {file_path}, using default zeros")
            fe_data = np.zeros(len(df))
        else:
            fe_data = df[fe_col].dropna().values
            if len(fe_data) == 0:
                fe_data = np.zeros(len(df))
        
        if not ba_col:
            print(f"Warning: BA column not found in {file_path}, using default zeros")
            ba_data = np.zeros(len(df))
        else:
            ba_data = df[ba_col].dropna().values
            if len(ba_data) == 0:
                ba_data = np.zeros(len(df))
        
        if not rpm_col:
            print(f"Warning: RPM column not found in {file_path}, using default 0")
            rpm = 0
        else:
            rpm_values = df[rpm_col].dropna().values
            if len(rpm_values) == 0:
                rpm = 0
            else:
                rpm = rpm_values[0]  # 获取转速值
        
        return de_data, fe_data, ba_data, rpm
        
    except Exception as e:
        print(f"Error loading data from {file_path}: {str(e)}")
        return None, None, None, None, None

def extract_time_domain_features(signal_data):
    """
    提取单通道时域特征
    """
    n = len(signal_data)
    if n == 0:
        return [0, 0, 0, 0, 0]
    
    # 检查信号是否全零
    if np.all(signal_data == 0):
        return [0, 0, 0, 0, 0]
    mean_val = np.mean(signal_data)
    rms_val = np.sqrt(np.mean(signal_data**2))
    peak_val = np.max(np.abs(signal_data))

    # 计算峰度和偏度时避免除零错误
    std_val = np.std(signal_data)
    if std_val == 0:
        kurtosis_val = 0
        skewness_val = 0
    else:
        kurtosis_val = np.mean((signal_data - mean_val)**4) / (std_val**4)
        skewness_val = np.mean((signal_data - mean_val)**3) / (std_val**3)
    
    return [mean_val, rms_val, peak_val, kurtosis_val, skewness_val]

def extract_frequency_domain_features(signal_data, fs):
    """
    提取单通道频域特征
    """
    n = len(signal_data)
    if n == 0 or np.all(signal_data == 0):
        return [0]
    fft_vals = np.abs(fft(signal_data))[:n//2]
    freqs = np.fft.fftfreq(n, 1/fs)[:n//2]
    
    if np.sum(fft_vals) == 0:
        spectral_centroid = 0
    else:
        spectral_centroid = np.sum(freqs * fft_vals) / np.sum(fft_vals)
    
    return [spectral_centroid]

def extract_wavelet_features(signal_data, wavelet='db4', level=3):
    """
    提取小波包能量特征，处理零值情况
    """
    if len(signal_data) == 0 or np.all(signal_data == 0):
        # 返回适当长度的零列表
        wp = pywt.WaveletPacket(data=np.zeros(100), wavelet=wavelet, mode='symmetric', maxlevel=level)
        nodes = [node.path for node in wp.get_level(level, 'natural')]
        return [0] * len(nodes)
    
    try:
        wp = pywt.WaveletPacket(data=signal_data, wavelet=wavelet, mode='symmetric', maxlevel=level)
        nodes = [node.path for node in wp.get_level(level, 'natural')]
        
        energy_list = []
        for node in nodes:
            coeff = wp[node].data
            energy = np.sum(coeff**2)
            energy_list.append(energy)
        
        energy_total = np.sum(energy_list)
        if energy_total == 0:
            energy_percent = [0] * len(energy_list)
        else:
            energy_percent = [e / energy_total for e in energy_list]
        
        return energy_percent
    except Exception as e:
        print(f"Error in wavelet feature extraction: {e}")
        # 返回适当长度的零列表
        wp = pywt.WaveletPacket(data=np.zeros(100), wavelet=wavelet, mode='symmetric', maxlevel=level)
        nodes = [node.path for node in wp.get_level(level, 'natural')]
        return [0] * len(nodes)

def calculate_correlation(signal1, signal2):
    """
    计算两个信号的相关系数
    """
    if len(signal1) != len(signal2):
        min_len = min(len(signal1), len(signal2))
        signal1 = signal1[:min_len]
        signal2 = signal2[:min_len]
    
    # 检查是否全零
    if np.all(signal1 == 0) or np.all(signal2 == 0):
        return 0
    
    # 检查方差是否为0（常数信号）
    if np.std(signal1) == 0 or np.std(signal2) == 0:
        return 0
    
    try:
        return np.corrcoef(signal1, signal2)[0, 1]
    except:
        return 0

# ---- 几何频率 + 对齐指标（包络谱上） ----
def bearing_freqs(fr_hz: float, Nd: int, d: float, D: float) -> dict:
    rho = d / D
    ftf  = 0.5 * (1 - rho) * fr_hz
    bpfo = 0.5 * Nd * (1 - rho) * fr_hz
    bpfi = 0.5 * Nd * (1 + rho) * fr_hz
    bsf  = (1 - rho**2) / (2*rho) * fr_hz
    return {"fr": fr_hz, "FTF": ftf, "BPFO": bpfo, "BPFI": bpfi, "BSF": bsf, "rho": rho}

def band_metrics(freqs, mag, f0, delta=2.0):
    idx = np.where((freqs >= f0 - delta) & (freqs <= f0 + delta))[0]
    if idx.size==0: return 0.0, 0.0
    peak = float(mag[idx].max())
    df = float(freqs[1]-freqs[0]) if len(freqs)>1 else 1.0
    energy = float((mag[idx]**2).sum() * df)
    return peak, energy

def harmonic_energy(freqs, mag, f0, M=5, delta=2.0):
    e=0.0
    for m in range(1, M+1):
        _, ei = band_metrics(freqs, mag, m*f0, delta)
        e += ei
    return e

def sideband_energy(freqs, mag, f0, fr, M=5, Q=3, delta=2.0):
    e=0.0
    for m in range(1, M+1):
        base = m*f0
        for q in range(1, Q+1):
            for sign in (-1, +1):
                _, ei = band_metrics(freqs, mag, base + sign*q*fr, delta)
                e += ei
    return e

def freq_aligned_indicators(env_mag, freqs, fr, targets: dict,
                            delta=2.0, M=5, Q=3, prefix=""):
    total_energy = float((env_mag**2).sum() * (freqs[1]-freqs[0] if len(freqs)>1 else 1.0))
    out = {}
    for key in ["FTF","BPFO","BPFI","BSF"]:
        f0 = targets[key]
        pk, be = band_metrics(freqs, env_mag, f0, delta)
        he = harmonic_energy(freqs, env_mag, f0, M, delta)
        sb = sideband_energy(freqs, env_mag, f0, fr, M, Q, delta)
        out[f"{prefix}{key}_peak"] = pk
        out[f"{prefix}{key}_bandE"] = be
        out[f"{prefix}{key}_Eratio"] = be / (total_energy + 1e-12)
        out[f"{prefix}{key}_harmE_M{M}"] = he
        out[f"{prefix}{key}_harmRatio_M{M}"] = he / (total_energy + 1e-12)
        out[f"{prefix}{key}_SB_Q{Q}"] = sb
        out[f"{prefix}{key}_SBI_Q{Q}"] = sb / (he + 1e-12)
    return out

def to_orders(freqs_hz: np.ndarray, fr_hz: float) -> np.ndarray:
    """Hz 频率轴 -> 阶次轴（orders）。"""
    fr = max(fr_hz, 1e-9)
    return freqs_hz / fr

def order_band_metrics(orders, mag, o0, delta_o=0.1):
    """阶次窗口内峰值与能量。"""
    idx = np.where((orders >= o0 - delta_o) & (orders <= o0 + delta_o))[0]
    if idx.size==0: return 0.0, 0.0
    peak = float(mag[idx].max())
    # 阶次轴“步长”用于近似积分（这里用均匀网格近似）
    do = float(orders[1]-orders[0] if len(orders)>1 else 1.0)
    energy = float((mag[idx]**2).sum() * do)
    return peak, energy

def order_harmonic_energy(orders, mag, o0, M=5, delta_o=0.1):
    e=0.0
    for m in range(1, M+1):
        _, ei = order_band_metrics(orders, mag, m*o0, delta_o)
        e += ei
    return e

def order_sideband_energy(orders, mag, o0, M=5, Q=3, delta_o=0.1):
    """
    在阶次域，调制的旁带间距 = ± q * 1阶（即 ± q）。
    这里以“以转频为调制”的常见情形计算旁带能量。
    """
    e=0.0
    for m in range(1, M+1):
        base = m*o0
        for q in range(1, Q+1):
            for sign in (-1, +1):
                _, ei = order_band_metrics(orders, mag, base + sign*q*1.0, delta_o)
                e += ei
    return e

def order_aligned_indicators(env_mag, freqs_hz, fr_hz, targets_hz: dict,
                             delta_o=0.1, M=5, Q=3, prefix=""):
    """
    把包络谱从 Hz -> 阶次，再在阶次轴上对齐 FTF/BPFO/BPFI/BSF 计算整段指标。
    """
    orders = to_orders(freqs_hz, fr_hz)
    do = float(orders[1]-orders[0] if len(orders)>1 else 1.0)
    total_energy_o = float((env_mag**2).sum() * do)

    # 目标频率（Hz）也转为目标阶次中心
    o_targets = {k: (targets_hz[k] / max(fr_hz,1e-9)) for k in ["FTF","BPFO","BPFI","BSF"]}

    out = {}
    for key, o0 in o_targets.items():
        pk, be = order_band_metrics(orders, env_mag, o0, delta_o)
        he = order_harmonic_energy(orders, env_mag, o0, M, delta_o)
        sb = order_sideband_energy(orders, env_mag, o0, M, Q, delta_o)
        out[f"{prefix}{key}_peak_ord"] = pk
        out[f"{prefix}{key}_bandE_ord"] = be
        out[f"{prefix}{key}_Eratio_ord"] = be / (total_energy_o + 1e-12)
        out[f"{prefix}{key}_harmE_M{M}_ord"] = he
        out[f"{prefix}{key}_harmRatio_M{M}_ord"] = he / (total_energy_o + 1e-12)
        out[f"{prefix}{key}_SB_Q{Q}_ord"] = sb
        out[f"{prefix}{key}_SBI_Q{Q}_ord"] = sb / (he + 1e-12)
    return out

def envelope_and_spectrum(x: np.ndarray, fs: int):
    analytic = signal.hilbert(x)
    env = np.abs(analytic)
    e = env - np.mean(env)
    X = np.fft.rfft(e)
    mag = np.abs(X)
    freqs = np.fft.rfftfreq(len(e), d=1/fs)
    return env, mag, freqs

def extract_features_from_data(de_data, fe_data, ba_data, rpm, fs=12000):
    """
    从三通道数据中提取完整特征向量
    """
    features = []
    feature_dict = {}  # 用于存储特征名称和值的字典
    # 1. 单通道时域特征
    features.extend(extract_time_domain_features(de_data))
    features.extend(extract_time_domain_features(fe_data))
    features.extend(extract_time_domain_features(ba_data))
    
    # 2. 通道间相关系数
    features.append(calculate_correlation(de_data, fe_data))
    features.append(calculate_correlation(de_data, ba_data))
    features.append(calculate_correlation(fe_data, ba_data))
    
    # 3. 单通道频域特征
    features.extend(extract_frequency_domain_features(de_data, fs))
    features.extend(extract_frequency_domain_features(fe_data, fs))
    features.extend(extract_frequency_domain_features(ba_data, fs))
    
    # 4. 小波包能量特征
    features.extend(extract_wavelet_features(de_data))
    features.extend(extract_wavelet_features(fe_data))
    features.extend(extract_wavelet_features(ba_data))
    
    # 5. 通道间能量比特征（小波包）
    de_wavelet = extract_wavelet_features(de_data)
    fe_wavelet = extract_wavelet_features(fe_data)
    ba_wavelet = extract_wavelet_features(ba_data)
    
    for i in range(len(de_wavelet)):
        if fe_wavelet[i] != 0:
            features.append(de_wavelet[i] / fe_wavelet[i])
        else:
            features.append(0)
        
        if ba_wavelet[i] != 0:
            features.append(de_wavelet[i] / ba_wavelet[i])
        else:
            features.append(0)
    
    # 添加转速信息
    features.append(rpm)

    # 几何频率（仅对DE和FE通道）
    fr_hz = rpm / 60.0  # 转换为Hz

    # 处理DE通道
    if np.isfinite(fr_hz) and fr_hz > 0:
        # 计算包络谱 
        _, env_mag_de, fvec = envelope_and_spectrum(de_data, TARGET_FS)
        _, env_mag_fe, fvec = envelope_and_spectrum(fe_data, TARGET_FS)
        # DE通道
        Nd = GEOM["DE"]["Nd"]; d = GEOM["DE"]["d"]; D = GEOM["DE"]["D"]
        geom_de = bearing_freqs(fr_hz, Nd, d, D)
        
        # 记录几何频率值
        feature_dict["DE_FTF"] = geom_de["FTF"]
        feature_dict["DE_BPFO"] = geom_de["BPFO"]
        feature_dict["DE_BPFI"] = geom_de["BPFI"]
        feature_dict["DE_BSF"] = geom_de["BSF"]
        feature_dict["DE_rho_d_over_D"] = geom_de["rho"]
        
         # >>> 新增：阶次域一致性（不需要窗口，整段） <<<
        # 固定阶次半宽 delta_o；你可按经验改 0.05~0.2
        aligned_ord_de =  order_aligned_indicators(env_mag_de, fvec, fr_hz,
                                                geom_de,delta_o=0.1, M=5, Q=3, prefix="DE")
        feature_dict.update(aligned_ord_de)

        # FE通道
        Nd = GEOM["FE"]["Nd"]; d = GEOM["FE"]["d"]; D = GEOM["FE"]["D"]
        geom_fe = bearing_freqs(fr_hz, Nd, d, D)
        
        feature_dict["FE_FTF"] = geom_fe["FTF"]
        feature_dict["FE_BPFO"] = geom_fe["BPFO"]
        feature_dict["FE_BPFI"] = geom_fe["BPFI"]
        feature_dict["FE_BSF"] = geom_fe["BSF"]
        feature_dict["FE_rho_d_over_D"] = geom_fe["rho"]

        aligned_ord_fe =  order_aligned_indicators(env_mag_fe, fvec, fr_hz,
                                        geom_fe,delta_o=0.1, M=5, Q=3, prefix="FE")
        feature_dict.update(aligned_ord_fe)

    # 将几何频率和对齐特征添加到特征向量中
    # 需要确保特征名称与create_feature_names函数中的顺序一致
    geometric_features = list(feature_dict.values())
    features.extend(geometric_features)

    return np.array(features)

def create_feature_names():
    """
    创建特征名称列表
    """
    feature_names = []
    
    # 时域特征名称
    channels = ['DE', 'FE', 'BA']
    time_features = ['Mean', 'RMS', 'Peak', 'Kurtosis', 'Skewness']
    for channel in channels:
        for feature in time_features:
            feature_names.append(f"{channel}_{feature}")
    
    # 相关系数特征名称
    feature_names.extend(['DE_FE_Correlation', 'DE_BA_Correlation', 'FE_BA_Correlation'])
    
    # 频域特征名称
    for channel in channels:
        feature_names.append(f"{channel}_SpectralCentroid")
    
    # 小波包能量特征名称
    wavelet_level = 3
    nodes = pywt.WaveletPacket(data=np.zeros(100), wavelet='db4', mode='symmetric', maxlevel=wavelet_level)
    node_names = [node.path for node in nodes.get_level(wavelet_level, 'natural')]
    
    for channel in channels:
        for node in node_names:
            feature_names.append(f"{channel}_WaveletEnergy_{node}")
    
    # 通道间能量比特征
    for node in node_names:
        feature_names.append(f"DE_FE_EnergyRatio_{node}")
        feature_names.append(f"DE_BA_EnergyRatio_{node}")
    
    # 转速特征
    feature_names.append('RPM')
    
    # 几何频率特征名称（新添加的）
    geometric_features = [
        "DE_FTF", "DE_BPFO", "DE_BPFI", "DE_BSF", "DE_rho_d_over_D",
        "FE_FTF", "FE_BPFO", "FE_BPFI", "FE_BSF", "FE_rho_d_over_D"
    ]
    feature_names.extend(geometric_features)
    
    # 阶次域对齐特征名称（新增）
    bearing_fault_types = ["FTF", "BPFO", "BPFI", "BSF"]
    metric_types = [
        "peak_ord",           # 峰值
        "bandE_ord",          # 频带能量
        "Eratio_ord",         # 能量比
        "harmE_M5_ord",       # 谐波能量（M=5）
        "harmRatio_M5_ord",   # 谐波能量比
        "SB_Q3_ord",          # 边带能量（Q=3）
        "SBI_Q3_ord",         # 边带指数
    ]
    
    # DE通道的阶次域特征
    for fault_type in bearing_fault_types:
        for metric in metric_types:
            feature_names.append(f"DE_{fault_type}_{metric}")
    
    # FE通道的阶次域特征
    for fault_type in bearing_fault_types:
        for metric in metric_types:
            feature_names.append(f"FE_{fault_type}_{metric}")

    return feature_names

def butter_bandpass(low, high, fs, order=4):
    nyq = 0.5*fs
    low_n, high_n = max(1e-9, low/nyq), min(0.999999, high/nyq)
    if high_n <= low_n: high_n = min(0.999999, low_n*1.5)
    b, a = signal.butter(order, [low_n, high_n], btype='band')
    return b, a

def infer_fs_from_path(path: Path, default_fs=DEFAULT_FS_GUESS) -> int:
    s = str(path).lower()
    if re.search(r"(48k|48000)", s): return 48000
    if re.search(r"(12k|12000)", s): return 12000
    return default_fs

def preprocess_whole(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    x = signal.detrend(np.asarray(x, dtype=float), type="linear")
    b, a = butter_bandpass(BP_LOW, min(BP_HIGH, 0.49*fs_in), fs=fs_in, order=FILTER_ORDER)
    x = signal.filtfilt(b, a, x, method="gust")
    g = math.gcd(fs_in, fs_out)
    up, down = fs_out//g, fs_in//g
    return signal.resample_poly(x, up=up, down=down, padtype="line")

# ---- 文件名标签解析 ----
LABEL_RE = re.compile(
    r"(?P<cls>OR|IR|B|N)"
    r"(?P<size>\d{3})?"
    r"(?:@(?P<pos>(3|6|12)))?"
    r"(?:_(?P<load>\d))?",
    re.IGNORECASE
)

def parse_label_from_name(path: Path) -> dict:
    name = path.stem.upper()
    m = LABEL_RE.search(name)
    out = {"cls": None, "size_in": None, "load_hp": None, "or_pos": None}
    if m:
        cls = m.group("cls").upper()
        out["cls"] = cls
        size = m.group("size")
        out["size_in"] = float(size)/1000.0 if (size and cls in {"OR","IR","B"}) else None
        ld = m.group("load")
        out["load_hp"] = int(ld) if ld is not None else None
        pos = m.group("pos")
        out["or_pos"] = int(pos) if (pos and cls=="OR") else None
    return out

def process_single_file(file_path,fs_in):
    """
    处理单个Excel文件并返回特征向量和文件标识
    """
    print(f"Processing file: {file_path}")
    
    # 从Excel文件读取数据
    de_data, fe_data, ba_data, rpm = load_data_from_excel(file_path)
    
    # 检查数据是否有效
    if de_data is None or fe_data is None or ba_data is None or rpm is None:
        print(f"Error: Missing data in the Excel file: {file_path}")
        return None, None
    
    # 0值处理
    de_data = np.array(de_data, dtype=np.float64)
    fe_data = np.array(fe_data, dtype=np.float64)
    ba_data = np.array(ba_data, dtype=np.float64)
    

    # 去除NaN值
    de_data = de_data[~np.isnan(de_data)]
    fe_data = fe_data[~np.isnan(fe_data)]   
    ba_data = ba_data[~np.isnan(ba_data)]
    
    # 检查数据长度
    if len(de_data) == 0 or len(fe_data) == 0 or len(ba_data) == 0:
        print(f"Error: No valid data in the Excel file: {file_path}")
        return None, None
    
    de_data = preprocess_whole(de_data, fs_in, TARGET_FS)
    fe_data = preprocess_whole(fe_data, fs_in, TARGET_FS)
    ba_data = preprocess_whole(ba_data, fs_in, TARGET_FS)
    
    # 数据截断或填充到相同长度
    min_length = min(len(de_data), len(fe_data), len(ba_data))
    de_data = de_data[:min_length]
    fe_data = fe_data[:min_length]
    ba_data = ba_data[:min_length]
    
    # 提取特征向量
    feature_vector = extract_features_from_data(de_data, fe_data, ba_data, rpm)
    
    return feature_vector

if __name__ == "__main__":
    parser = ArgumentParser(description="script parameters")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input Excel file or directory")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output Excel file")
    args = parser.parse_args(sys.argv[1:])

    # 获取所有以"merged"开头的Excel文件
    excel_files = []
    if os.path.isfile(args.input_path) and args.input_path.endswith(('.xlsx', '.xls')):
        file_name = os.path.basename(args.input_path)
        if file_name.lower().startswith('merged'):
            excel_files = [args.input_path]
        else:
            print(f"File {args.input_path} does not start with 'merged', skipping")
    elif os.path.isdir(args.input_path):
        # 查找所有以"merged"开头的Excel文件（不区分大小写）
        excel_files = glob.glob(os.path.join(args.input_path, "**", "merged*.xlsx"), recursive=True)
        excel_files.extend(glob.glob(os.path.join(args.input_path, "**", "merged*.xls"), recursive=True))
    else:
        print(f"Error: Input path {args.input_path} is not a valid file or directory")
        sys.exit(1)
    
    if not excel_files:
        print(f"No Excel files starting with 'merged' found in {args.input_path}")
        sys.exit(1)
    
    print(f"Found {len(excel_files)} Excel files starting with 'merged' to process")
    
    # 创建特征名称
    feature_names = create_feature_names()
    
    # 处理所有文件
    all_features = []

    labels = []
    for file_path in excel_files:
        fs_in = infer_fs_from_path(Path(file_path), DEFAULT_FS_GUESS) # 输入频率
        feature_vector = process_single_file(file_path,fs_in) # 特征向量
        if feature_vector is not None:
            all_features.append(feature_vector)
            # 当前样本的标签
            meta = parse_label_from_name(Path(file_path))
            label = {
                "File Name": str(os.path.basename(file_path)),
                "fs_inferred": fs_in,
                "fs_target": TARGET_FS,
                "cls": meta["cls"],
                "size_in": meta["size_in"],
                "load_hp": meta["load_hp"],
            }
            labels.append(label)
    
    if not all_features:
        print("No valid features extracted from any file")
        sys.exit(1)
    
    # 创建DataFrame
    feature_df = pd.DataFrame(all_features, columns=feature_names)
    # 将标签信息转换为DataFrame并合并到特征DataFrame中
    labels_df = pd.DataFrame(labels)
    feature_df = pd.concat([ labels_df,feature_df], axis=1)
    
    # 保存到csv文件
    feature_df.to_csv(args.output_path, index=False)
    
    print(f"Successfully processed {len(all_features)} files")
    print(f"Output saved to: {args.output_path}")
    print(f"Feature vector dimension: {len(feature_names)}")