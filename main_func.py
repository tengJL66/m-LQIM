import cv2
import numpy as np
import librosa
import soundfile

from sync_main import extract_audio_from_mp3_with_ffmpeg, calculate_ncorr
from tool.LSC import lsc_attack
from tool.LFSR import *
from tool.attack_methods import *
from tool.metric import calculate_odg
from tool.ss_watermark import *
from tool.wm_extraction import *
from tool.qim_watermark import *
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

if __name__ == '__main__':
    odg_list = []
    mse_list = []
    ber_list = []
    corr_list = []

    wm_type = 'mSWLSC'
    for name in range(1):
        # 读取水印信息
        W_ori = cv2.imread('watermark.bmp', cv2.IMREAD_GRAYSCALE)
        W = np.asarray(W_ori, dtype=int)
        W[W == 255] = 1
        W = W.flatten()

        # 加载音频文件
        audio_path = 'audio_file/disco00' + str(name) + '.wav'
        signal, sr = librosa.load(audio_path, sr=None)
        # 嵌入水印
        _lambda = 0.5
        if wm_type == "GBTSVD":
            wm_signal = GS_watermark_embed(signal, W, _lambda)
        else:
            if wm_type=="mSWLSC":
                wm_signal = ss_watermark_embed(signal, W, _lambda)


        # b = _lambda * Lattice_Basis('Z',1)
        # wm_signal = qim_watermark_embed(signal, W,b,_lambda)
        print("mse", np.mean((signal - wm_signal) ** 2))


        wm_signal_len = len(W)*100
        # 一般方式-提取水印 ss水印
        if wm_type == "GBTSVD":
            w_extract =  GS_watermark_extract(wm_signal, signal, W,_lambda)
        else:
            if wm_type=="mSWLSC":
                w_extract = ss_watermark_extract(wm_signal, signal, W,_lambda)

        soundfile.write("audio_file/wm_signal_ms.wav", wm_signal, sr)


        # 各种攻击
        # GA
        # attack_name = "GS_GA20"
        # wm_signal[0:wm_signal_len] = add_gaussian_noise(wm_signal[0:wm_signal_len],10)

        # MP3v
        #
        # wm_signal_64k = "audio_file/wm_signal_ms_64k.mp3"
        # wm_signal_128k = "audio_file/wm_signal_ms_128k.mp3"
        # compress_mp3("audio_file/wm_signal_ms.wav",wm_signal_64k,"64K")
        # compress_mp3("audio_file/wm_signal_ms.wav", wm_signal_128k, "128K")
        # wm_signal_64k_signal = extract_audio_from_mp3_with_ffmpeg(wm_signal_64k)
        # wm_signal_128k_signal = extract_audio_from_mp3_with_ffmpeg(wm_signal_128k)
        # wm_signal = np.copy(wm_signal_128k_signal)
        # 缩放攻击
        # wm_signal[0:wm_signal_len] = np.copy(wm_signal[0:wm_signal_len]) * 0.8

        # 重采样攻击 同步头
        # attack_name = "GS_rs"
        # rs_signal = resample_audio(wm_signal,sr,44100)
        # wm_signal = resample_audio(rs_signal,44100,22050)

        # 低通滤波 同步头

        # cutoff_frequency = 2000  # 截止频率 3000 Hz
        # filter_order = 5  # 滤波器阶数
        # 应用低通滤波器
        # filtered_audio = butter_lowpass_filter(wm_signal, cutoff_frequency, sr, filter_order)
        # wm_signal = filtered_audio

        # 中值滤波 同步头
        # attack_name = "mlsc_median"
        # wm_signal = median_filter(wm_signal, kernel_size=5)

        # 重量化攻击 无法抵抗 同步头？
        # 原始音频是16位的，我们将其重量化到8位，然后再重量化回16位
        # audio_8bit = requantize(wm_signal, 16, 8)
        # wm_signal = requantize(audio_8bit, 8, 16)

        # LSC攻击
        lsc_signal, crop_starts = lsc_attack(wm_signal, 0.9, 10)
        print("裁剪位置：", crop_starts)
        # lsc_signal = wm_signal

        # w_extract = GS_watermark_extract(wm_signal, signal, W)
        # 再次重塑为50x50
        data_reshaped = np.asarray(w_extract)
        matrix = data_reshaped.reshape(50, 50)
        matrix[matrix == 1] = 255
        # 将NumPy数组转换为图像
        matrix = matrix.astype(np.uint8)
        # attack_name = attack_name + '_image.bmp'
        # cv2.imwrite(attack_name, matrix)

        # 计算错误比特数
        error_bits = np.sum(matrix != W_ori)
        # 计算总比特数
        total_bits = matrix.size
        # 计算误码率
        ber = error_bits / total_bits
        attack_name = "AM"
        print(attack_name, "水印BER:", ber)
        print(attack_name, "水印corr:", calculate_ncorr(W, matrix.flatten()))

        odg_value = calculate_odg(signal[0:wm_signal_len], wm_signal[0:wm_signal_len], sr)
        mse_list.append(np.mean((signal[0:wm_signal_len] - wm_signal[0:wm_signal_len]) ** 2))
        odg_list.append(odg_value)
        ber_list.append(ber)
        corr_list.append(np.corrcoef(W, matrix.flatten())[0, 1])

    # print("odg", odg_list)
    # print(sum(odg_list) / 10)
    #
    # print("mse", mse_list)
    # print(sum(mse_list) / 10)
    #
    # print("ber", ber_list)
    # print(sum(ber_list) / 10)
    #
    # print("corr", corr_list)
    # print(sum(corr_list) / 10)

# """是
    
    # 生成水印密钥
    # 本原多项式
    L = 13
    polynomial = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]  # 注意：多项式从高到低表示 L=13
    # 初始状态 [0, 0, 0, 1]
    initial_state = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    # 创建 LFSR
    lfsr = LFSR(polynomial, initial_state)
    # 生成 15 位伪随机序列（最大周期为 2^4 - 1 = 15）
    sequence = lfsr.generate_sequence(2 ** L - 1)
    print("生成的伪随机序列长度:", len(sequence))
    Wkey = W ^ sequence[0:len(W)]

    # m-提取
    extracted_watermark = ss_m_watermark_extraction(lsc_signal, signal, Wkey, _lambda,sequence[0:len(W)])
    W_re = np.asarray(extracted_watermark)

    # 再次重塑为50x50
    data_reshaped = np.asarray(W_re)
    matrix = data_reshaped.reshape(50, 50)
    matrix[matrix == 1] = 255
    # 将NumPy数组转换为图像
    matrix = matrix.astype(np.uint8)
    attack_name = attack_name + '_image.bmp'
    # cv2.imwrite(attack_name, matrix)
    W_re = W_re.reshape(W_ori.shape)

    # 计算错误比特数
    error_bits = np.sum(matrix != W_ori)
    # 计算总比特数
    total_bits = matrix.size
    # 计算误码率
    ber = error_bits / total_bits
    print(attack_name, "m提取水印BER:", ber)
    print(attack_name, "m提取水印corr:", calculate_ncorr(W,matrix.flatten()))


# """
