import numpy as np
import librosa


def calculate_snr(signal, noise):
    """
    计算信噪比 (SNR)
    :param signal: 信号数组
    :param noise: 噪声数组
    :return: SNR (dB)
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def calculate_odg(original_signal, processed_signal, sr):
    """
    计算两个音频信号之间的 ODG 值
    :param original_signal: 原始音频信号
    :param processed_signal: 处理后的音频信号
    :param sr: 采样率
    :return: ODG 值
    """
    # 计算短时能量
    def short_time_energy(signal, frame_size, hop_length):
        frames = librosa.util.frame(signal, frame_length=frame_size, hop_length=hop_length)
        return np.sum(frames ** 2, axis=0)

    # 计算短时能量的均值和标准差
    frame_size = 1024
    hop_length = 512

    original_energy = short_time_energy(original_signal, frame_size, hop_length)
    processed_energy = short_time_energy(processed_signal, frame_size, hop_length)

    original_mean = np.mean(original_energy)
    original_std = np.std(original_energy)
    processed_mean = np.mean(processed_energy)
    processed_std = np.std(processed_energy)

    # 计算 ODG
    odg = 10 * np.log10((original_mean + original_std) / (processed_mean + processed_std))

    return odg

