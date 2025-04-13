from pydub import AudioSegment
import numpy as np
from scipy import ndimage
from scipy.io import wavfile
from scipy.signal import butter, lfilter


def compress_mp3(input_file, output_file, bitrate):
    """
    压缩 MP3 文件到指定比特率
    :param input_file: 输入的 MP3 文件路径
    :param output_file: 输出的 MP3 文件路径
    :param bitrate: 压缩后的比特率（如 '64k' 或 '128k'）
    """
    # 加载 WAV 文件
    wav_audio = AudioSegment.from_file(input_file, format="wav")

    # 导出为 MP3 文件，指定比特率
    wav_audio.export(output_file, format="mp3", bitrate=bitrate)


def add_gaussian_noise(signal, snr_db):
    """
    给信号添加加性高斯白噪声
    :param signal: 输入信号，numpy数组
    :param snr_db: 信噪比，单位为dB
    :return: 加噪后的信号
    """
    # 计算信号功率
    signal_power = np.mean(signal ** 2)
    # 将信噪比从dB转换为线性比例
    snr_linear = 10 ** (snr_db / 10)
    # 计算噪声功率
    noise_power = signal_power / snr_linear
    # 生成高斯白噪声
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    # 将噪声添加到信号中
    noisy_signal = signal + noise
    return noisy_signal


def resample_audio(audio_data, original_sample_rate, new_sample_rate):
    """
    对音频信号进行重采样攻击
    :param input_file: 输入的音频文件路径
    :param output_file: 输出的音频文件路径
    :param new_sample_rate: 新的采样率
    """

    # 计算时间轴
    t = np.arange(len(audio_data)) / original_sample_rate

    # 重采样
    new_t = np.arange(0, t[-1], 1 / new_sample_rate)
    new_audio_data = np.interp(new_t, t, audio_data)
    return new_audio_data


def butter_lowpass(cutoff, fs, order=5):
    """
    设计巴特沃斯低通滤波器
    :param cutoff: 截止频率
    :param fs: 采样频率
    :param order: 滤波器阶数
    :return: 滤波器的分子和分母多项式系数
    """
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    应用巴特沃斯低通滤波器
    :param data: 输入信号
    :param cutoff: 截止频率
    :param fs: 采样频率
    :param order: 滤波器阶数
    :return: 滤波后的信号
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def median_filter(data, kernel_size):
    """
    应用中值滤波
    :param data: 输入信号或图像
    :param kernel_size: 滤波器窗口大小
    :return: 滤波后的信号或图像
    """
    # 使用 scipy.ndimage.median_filter 实现中值滤波
    return ndimage.median_filter(data, size=kernel_size)


def requantize(audio_data, original_bits, new_bits):
    """
    对音频数据进行重量化处理
    :param audio_data: 原始音频数据
    :param original_bits: 原始量化位数
    :param new_bits: 新的量化位数
    :return: 重量化后的音频数据
    """
    # 计算原始和新的量化级别
    original_levels = 2 ** original_bits
    new_levels = 2 ** new_bits

    # 缩放音频数据到[0, 1]范围
    audio_scaled = audio_data / (original_levels - 1)

    # 重量化到新的级别
    audio_quantized = np.round(audio_scaled * (new_levels - 1))

    # 缩放回原始范围
    audio_requantized = (audio_quantized / (new_levels - 1)) * (original_levels - 1)

    return audio_requantized

