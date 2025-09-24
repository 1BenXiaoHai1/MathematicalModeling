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

# 目标域数据参数
TARGET_FS = 32000  # 采样频率32kHz
TARGET_RPM = 600   # 轴承转速约600rpm
BP_LOW, BP_HIGH = 500.0, 10000.0  # 带通滤波带宽
FILTER_ORDER = 4

# 源域轴承几何参数（保持与源域一致）
GEOM = {
    "DE": {"Nd": 9, "d": 0.3126, "D": 1.537},  # SKF6205
    "FE": {"Nd": 9, "d": 0.2656, "D": 1.122},  # SKF6203
}    

def load_target_data_from_excel(file_path):
    """
    从Excel文件读取目标域振动数据
    假设Excel文件中包含振动数据列
    """
    try:
        df = pd.read_excel(file_path)
        
        # 自动识别振动数据列（假设是第一列数值数据）
        vibration_col = None
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                vibration_data = df[col].dropna().values
                if len(vibration_data) > 1000:  # 确保是振动数据而不是标签
                    vibration_col = col
                    break
        
        if vibration_col is None:
            # 如果没有找到合适的列，使用第一列数值数据
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    vibration_col = col
                    break
        
        if vibration_col is None:
            print(f"Warning: No numeric data found in {file_path}")
            return None
        
        vibration_data = df[vibration_col].dropna().values
        
        # 检查数据长度（8秒，32kHz采样率应该有256000个点）
        expected_length = 8 * TARGET_FS  # 256000
        if len(vibration_data) < expected_length:
            print(f"Warning: Data length {len(vibration_data)} is less than expected {expected_length}")
            # 使用实际长度
        elif len(vibration_data) > expected_length:
            # 截取前8秒数据
            vibration_data = vibration_data[:expected_length]
        
        return vibration_data
        
    except Exception as e:
        print(f"Error loading target data from {file_path}: {str(e)}")
        return None

def extract_time_domain_features(signal_data):
    """
    提取单通道时域特征
    """
    n = len(signal_data)
    if n == 0:
        return [0, 0, 0, 0, 0]
    
    if np.all(signal_data == 0):
        return [0, 0, 0, 0, 0]
    
    mean_val = np.mean(signal_data)
    rms_val = np.sqrt(np.mean(signal_data**2))
    peak_val = np.max(np.abs(signal_data))

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
    提取小波包能量特征
    """
    if len(signal_data) == 0 or np.all(signal_data == 0):
        wp = pywt.WaveletPacket(data=np.zeros(100), wavelet=wavelet, mode='symmetric', maxlevel=level)
        nodes = [node.path for node in wp.get_level(level, 'natural')]
        return [0] * len(nodes)
    
    try:
        # 如果数据太长，可以适当下采样以提高计算效率
        if len(signal_data) > 10000:
            signal_data = signal.resample(signal_data, 10000)
        
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
    
    if np.all(signal1 == 0) or np.all(signal2 == 0):
        return 0
    
    if np.std(signal1) == 0 or np.std(signal2) == 0:
        return 0
    
    try:
        return np.corrcoef(signal1, signal2)[0, 1]
    except:
        return 0

def bearing_freqs(fr_hz: float, Nd: int, d: float, D: float) -> dict:
    """
    计算轴承故障特征频率
    """
    rho = d / D
    ftf  = 0.5 * (1 - rho) * fr_hz
    bpfo = 0.5 * Nd * (1 - rho) * fr_hz
    bpfi = 0.5 * Nd * (1 + rho) * fr_hz
    bsf  = (1 - rho**2) / (2*rho) * fr_hz
    return {"fr": fr_hz, "FTF": ftf, "BPFO": bpfo, "BPFI": bpfi, "BSF": bsf, "rho": rho}

def band_metrics(freqs, mag, f0, delta=2.0):
    """
    频带指标计算
    """
    idx = np.where((freqs >= f0 - delta) & (freqs <= f0 + delta))[0]
    if idx.size == 0: 
        return 0.0, 0.0
    peak = float(mag[idx].max())
    df = float(freqs[1]-freqs[0]) if len(freqs)>1 else 1.0
    energy = float((mag[idx]**2).sum() * df)
    return peak, energy

def harmonic_energy(freqs, mag, f0, M=5, delta=2.0):
    """
    谐波能量计算
    """
    e = 0.0
    for m in range(1, M+1):
        _, ei = band_metrics(freqs, mag, m*f0, delta)
        e += ei
    return e

def sideband_energy(freqs, mag, f0, fr, M=5, Q=3, delta=2.0):
    """
    边带能量计算
    """
    e = 0.0
    for m in range(1, M+1):
        base = m*f0
        for q in range(1, Q+1):
            for sign in (-1, +1):
                _, ei = band_metrics(freqs, mag, base + sign*q*fr, delta)
                e += ei
    return e

def to_orders(freqs_hz: np.ndarray, fr_hz: float) -> np.ndarray:
    """Hz频率轴转换为阶次轴"""
    fr = max(fr_hz, 1e-9)
    return freqs_hz / fr

def order_band_metrics(orders, mag, o0, delta_o=0.1):
    """阶次窗口内峰值与能量"""
    idx = np.where((orders >= o0 - delta_o) & (orders <= o0 + delta_o))[0]
    if idx.size == 0: 
        return 0.0, 0.0
    peak = float(mag[idx].max())
    do = float(orders[1]-orders[0]) if len(orders)>1 else 1.0
    energy = float((mag[idx]**2).sum() * do)
    return peak, energy

def order_harmonic_energy(orders, mag, o0, M=5, delta_o=0.1):
    """阶次谐波能量"""
    e = 0.0
    for m in range(1, M+1):
        _, ei = order_band_metrics(orders, mag, m*o0, delta_o)
        e += ei
    return e

def order_sideband_energy(orders, mag, o0, M=5, Q=3, delta_o=0.1):
    """阶次边带能量"""
    e = 0.0
    for m in range(1, M+1):
        base = m*o0
        for q in range(1, Q+1):
            for sign in (-1, +1):
                _, ei = order_band_metrics(orders, mag, base + sign*q*1.0, delta_o)
                e += ei
    return e

def order_aligned_indicators(env_mag, freqs_hz, fr_hz, targets_hz: dict, delta_o=0.1, M=5, Q=3, prefix=""):
    """阶次对齐指标计算"""
    orders = to_orders(freqs_hz, fr_hz)
    do = float(orders[1]-orders[0]) if len(orders)>1 else 1.0
    total_energy_o = float((env_mag**2).sum() * do)

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
    """包络谱分析"""
    # 如果数据太长，可以分段处理或下采样
    if len(x) > 100000:
        x = signal.resample(x, 100000)
        fs_effective = 100000 / (len(x) / fs)
    else:
        fs_effective = fs
    
    analytic = signal.hilbert(x)
    env = np.abs(analytic)
    e = env - np.mean(env)
    X = np.fft.rfft(e)
    mag = np.abs(X)
    freqs = np.fft.rfftfreq(len(e), d=1/fs_effective)
    return env, mag, freqs

def butter_bandpass(low, high, fs, order=4):
    """巴特沃斯带通滤波器"""
    nyq = 0.5 * fs
    low_n = max(1e-9, low/nyq)
    high_n = min(0.999999, high/nyq)
    if high_n <= low_n: 
        high_n = min(0.999999, low_n*1.5)
    b, a = signal.butter(order, [low_n, high_n], btype='band')
    return b, a

def preprocess_signal(x: np.ndarray, fs: int) -> np.ndarray:
    """信号预处理"""
    # 如果数据太长，可以适当下采样
    if len(x) > 100000:
        x = signal.resample(x, 100000)
    
    x = signal.detrend(np.asarray(x, dtype=float), type="linear")
    b, a = butter_bandpass(BP_LOW, min(BP_HIGH, 0.49*fs), fs=fs, order=FILTER_ORDER)
    x = signal.filtfilt(b, a, x, method="gust")
    return x

def extract_features_from_data(de_data, fe_data, ba_data, rpm, fs=TARGET_FS):
    """
    从三通道数据中提取完整特征向量（与源域完全一致）
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
    de_wavelet = extract_wavelet_features(de_data)
    fe_wavelet = extract_wavelet_features(fe_data)
    ba_wavelet = extract_wavelet_features(ba_data)
    
    features.extend(de_wavelet)
    features.extend(fe_wavelet)
    features.extend(ba_wavelet)
    
    # 5. 通道间能量比特征（小波包）
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

    # 几何频率（对DE和FE通道分别计算）
    fr_hz = rpm / 60.0  # 转换为Hz

    if np.isfinite(fr_hz) and fr_hz > 0:
        # 计算包络谱 
        _, env_mag_de, fvec = envelope_and_spectrum(de_data, fs)
        _, env_mag_fe, fvec_fe = envelope_and_spectrum(fe_data, fs)
        
        # DE通道（使用SKF6205参数）
        Nd_de = GEOM["DE"]["Nd"]; d_de = GEOM["DE"]["d"]; D_de = GEOM["DE"]["D"]
        geom_de = bearing_freqs(fr_hz, Nd_de, d_de, D_de)
        
        # 记录几何频率值
        feature_dict["DE_FTF"] = geom_de["FTF"]
        feature_dict["DE_BPFO"] = geom_de["BPFO"]
        feature_dict["DE_BPFI"] = geom_de["BPFI"]
        feature_dict["DE_BSF"] = geom_de["BSF"]
        feature_dict["DE_rho_d_over_D"] = geom_de["rho"]
        
        # 阶次域对齐特征
        aligned_ord_de = order_aligned_indicators(env_mag_de, fvec, fr_hz, geom_de, 
                                                delta_o=0.1, M=5, Q=3, prefix="DE_")
        feature_dict.update(aligned_ord_de)

        # FE通道（使用SKF6203参数）
        Nd_fe = GEOM["FE"]["Nd"]; d_fe = GEOM["FE"]["d"]; D_fe = GEOM["FE"]["D"]
        geom_fe = bearing_freqs(fr_hz, Nd_fe, d_fe, D_fe)
        
        feature_dict["FE_FTF"] = geom_fe["FTF"]
        feature_dict["FE_BPFO"] = geom_fe["BPFO"]
        feature_dict["FE_BPFI"] = geom_fe["BPFI"]
        feature_dict["FE_BSF"] = geom_fe["BSF"]
        feature_dict["FE_rho_d_over_D"] = geom_fe["rho"]

        aligned_ord_fe = order_aligned_indicators(env_mag_fe, fvec_fe, fr_hz, geom_fe,
                                                delta_o=0.1, M=5, Q=3, prefix="FE_")
        feature_dict.update(aligned_ord_fe)

    # 将几何频率和对齐特征添加到特征向量中
    geometric_features = list(feature_dict.values())
    features.extend(geometric_features)

    return np.array(features)

def create_feature_names():
    """
    创建特征名称列表（与源域完全一致）
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
    
    # 几何频率特征名称
    geometric_features = [
        "DE_FTF", "DE_BPFO", "DE_BPFI", "DE_BSF", "DE_rho_d_over_D",
        "FE_FTF", "FE_BPFO", "FE_BPFI", "FE_BSF", "FE_rho_d_over_D"
    ]
    feature_names.extend(geometric_features)
    
    # 阶次域对齐特征名称
    bearing_fault_types = ["FTF", "BPFO", "BPFI", "BSF"]
    metric_types = [
        "peak_ord", "bandE_ord", "Eratio_ord", 
        "harmE_M5_ord", "harmRatio_M5_ord", 
        "SB_Q3_ord", "SBI_Q3_ord"
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

def process_target_file(file_path, file_id):
    """
    处理单个目标域数据文件，将单通道数据当作DE、FE、BA三个通道处理
    """
    print(f"Processing target file: {file_path}")
    
    # 加载数据
    vibration_data = load_target_data_from_excel(file_path)
    
    if vibration_data is None:
        print(f"Error: Failed to load data from {file_path}")
        return None
    
    # 检查数据有效性
    if len(vibration_data) == 0:
        print(f"Error: No valid data in {file_path}")
        return None
    
    print(f"Loaded vibration data with {len(vibration_data)} points")
    
    # 预处理信号
    processed_data = preprocess_signal(vibration_data, TARGET_FS)
    
    # 将单通道数据复制为DE、FE、BA三个通道
    de_data = processed_data.copy()
    fe_data = processed_data.copy()  
    ba_data = processed_data.copy()
    
    # 提取特征（使用与源域完全相同的特征提取函数）
    feature_vector = extract_features_from_data(de_data, fe_data, ba_data, TARGET_RPM, TARGET_FS)
    
    # 创建结果字典
    result = {
        "File_Name": os.path.basename(file_path),
        "fs_inferred": TARGET_FS,
        "fs_target": TARGET_FS,
        "cls": "Unknown",  # 目标域故障类型未知
        "size_in": "Unknown",   # 目标域无故障尺寸信息
        "load_hp": "Unknown",   # 目标域无载荷信息
        "or_pos": "Unknown",    # 目标域无故障位置信息
    }
    
    return result, feature_vector

if __name__ == "__main__":
    parser = ArgumentParser(description="Target domain feature extraction with same features as source domain")
    parser.add_argument("--input_path", type=str, required=True, help="Path to target Excel files or directory")
    parser.add_argument("--output_path", type=str, required=True, help="Output CSV file path")
    args = parser.parse_args()
    
    # 获取目标Excel文件（A.xlsx, B.xlsx, ..., P.xlsx）
    target_files = []
    if os.path.isfile(args.input_path) and args.input_path.endswith(('.xlsx', '.xls')):
        target_files = [args.input_path]
    elif os.path.isdir(args.input_path):
        # 查找所有Excel文件，按A~P排序
        excel_files = glob.glob(os.path.join(args.input_path, "*.xlsx"))
        excel_files.extend(glob.glob(os.path.join(args.input_path, "*.xls")))
        
        # 按文件名排序（A.xlsx, B.xlsx, ..., P.xlsx）
        target_files = sorted(excel_files, key=lambda x: os.path.basename(x))
    else:
        print(f"Error: Input path {args.input_path} is not valid")
        sys.exit(1)
    
    if not target_files:
        print(f"No Excel files found in {args.input_path}")
        sys.exit(1)
    
    print(f"Found {len(target_files)} target Excel files")
    for i, file_path in enumerate(target_files):
        print(f"  {i+1}. {os.path.basename(file_path)}")
    
    # 创建特征名称（与源域完全一致）
    feature_names = create_feature_names()
    
    # 处理所有文件
    all_features = []
    file_info = []
    
    for i, file_path in enumerate(target_files):
        file_id = chr(65 + i)  # A, B, C, ..., P
        result = process_target_file(file_path, file_id)
        
        if result is not None:
            file_info_dict, feature_vector = result
            all_features.append(feature_vector)
            file_info.append(file_info_dict)
            print(f"Successfully processed {file_id}: {os.path.basename(file_path)}")
        else:
            print(f"Failed to process {file_id}: {os.path.basename(file_path)}")
    
    if not all_features:
        print("No valid features extracted from any target file")
        sys.exit(1)
    
    # 创建DataFrame（列顺序与源域完全一致）
    features_df = pd.DataFrame(all_features, columns=feature_names)
    info_df = pd.DataFrame(file_info)
    
    # 确保列顺序与源域一致：基本信息列在前，特征列在后
    result_df = pd.concat([info_df, features_df], axis=1)
    
    # 保存结果
    result_df.to_csv(args.output_path, index=False)
    
    print(f"\nSuccessfully processed {len(all_features)} target files")
    print(f"Output saved to: {args.output_path}")
    print(f"Feature vector dimension: {len(feature_names)}")