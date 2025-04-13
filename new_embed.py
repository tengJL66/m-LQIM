import cv2
import numpy as np
from tool import wm_extraction
import librosa
from lattice_embed.Lattice.Lattice_basis import Lattice_Basis
from tool import new_watermark
from tool.LFSR import LFSR
from tool.LSC import lsc_attack

if __name__ == '__main__':
    # 准备原信号和水印
    # 读取水印信息
    W_ori = cv2.imread('watermark.bmp', cv2.IMREAD_GRAYSCALE)
    W = np.asarray(W_ori, dtype=int)
    W[W == 255] = 1
    W = W.flatten()

    # 加载音频文件
    audio_path = 'D:/Mr.Liu/project/LSC-watermark/blues001.wav'
    signal, sr = librosa.load(audio_path, sr=None)
    seg_len = 100
    # 水印嵌入
    _lambda = 0.01
    b = _lambda * Lattice_Basis('Z', 2)
    wm_signal = new_watermark.qim_watermark_embed(signal, W, b, seg_len)
    print("水印嵌入mse", np.mean((signal[0:len(W)] - wm_signal[0:len(W)]) ** 2))

    # 定义高斯噪声的均值和标准差
    mean = 0  # 均值
    std_dev = 0.001  # 标准差

    # 生成与数组长度相同的高斯噪声
    noise = np.random.normal(mean, std_dev, (len(W) * seg_len,))
    wm_signal[0:len(noise)] += noise
    print("加噪mse", np.mean((signal[0:len(W)] - wm_signal[0:len(W)]) ** 2))


    # 水印提取
    w_extract = new_watermark.qim_watermark_extract(wm_signal, W, b,seg_len)

# """

    # LSC攻击
    lsc_signal, crop_starts = lsc_attack(wm_signal, 0.5, 10)
    print("裁剪位置：", crop_starts)

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
    extracted_watermark = wm_extraction.new_qim_m_watermark_extraction(lsc_signal,
                                           signal, Wkey, _lambda, b, W)

    W_re = np.asarray(extracted_watermark)
    W_re = W_re.reshape(W_ori.shape)
    W_ori[W_ori == 255] = 1
    if np.array_equal(W_re, W_ori):
        print("经裁切后，水印可以完整提取")
    else:
        # 计算错误比特数
        error_bits = np.sum(W_re != W_ori)
        # 计算总比特数
        total_bits = W_re.size
        # 计算误码率
        ber = error_bits / total_bits
        print("BER:", ber)
        print("经裁切后，水印无法完整提取")

    print("mse", np.mean((signal - wm_signal) ** 2))
# """
