from scipy.io import loadmat, savemat

from lattice_embed import Create_Data
from lattice_embed.QIM import cosets, standard_qim, qim_decode
from tool import new_watermark, wm_extraction
from tool.LFSR import LFSR
from lattice_embed.Lattice.Lattice_basis import Lattice_Basis
import math
import cv2
import numpy as np
import librosa
from scipy.signal import find_peaks

from tool.LSC import lsc_attack
from tool.attack_methods import *
import soundfile as sf
from multiprocessing import Pool
import time
from PIL import Image
import subprocess
from tool.metric import *


def extract_audio_from_mp3_with_ffmpeg(input_mp3_file):
    """
    使用 ffmpeg 从 MP3 文件中提取音频信号
    :param input_mp3_file: 输入的 MP3 文件路径
    :return: 音频信号（numpy 数组）
    """
    # 使用 ffmpeg 读取音频数据
    command = [
        "ffmpeg",
        "-i", input_mp3_file,
        "-f", "f32le",
        "-ac", "1",  # 单声道
        "-ar", "22050",  # 采样率 44.1kHz
        "-"
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio_data = np.frombuffer(process.stdout, dtype=np.float32)
    return audio_data


def detect_sync(audio_segment, sync_template, threshold):
    """
    检测音频段中的同步头
    :param audio_segment: 音频段
    :param sync_template: 同步头模板
    :param threshold: 检测阈值
    :return: 检测到的同步头位置
    """
    # 计算音频段与同步模板的互相关
    correlation = np.correlate(audio_segment, sync_template, mode='same')
    peaks, _ = find_peaks(correlation, height=threshold)
    return peaks


# 循环移位判断
def is_cyclic_shift(seq1, seq2):
    if len(seq1) != len(seq2):
        return False
    return any(seq1[i:] + seq1[:i] == seq2 for i in range(len(seq1)))


def find_sequence(array, target_sequence):
    """
    在数组中查找特定序列的位置
    :param array: 输入数组
    :param target_sequence: 目标序列
    :return: 目标序列的起始位置（找到时返回索引，未找到时返回-1）
    """
    target_length = len(target_sequence)
    array_length = len(array)

    # 遍历数组，使用滑动窗口查找目标序列
    for i in range(array_length - target_length + 1):
        if np.array_equal(array[i:i + target_length], target_sequence):
            return i  # 找到目标序列，返回起始索引
    return -1  # 未找到目标序列，返回-1




def calculate_ncorr(W, W_prime):
    """
    计算归一化相关系数 (NC)。

    参数:
    W (numpy.ndarray): 原始水印图像 (二维数组)
    W_prime (numpy.ndarray): 提取的水印图像 (二维数组)

    返回:
    float: 归一化相关系数 (NC)
    """
    # 确保输入是二维数组
    if W.ndim != 2 or W_prime.ndim != 2:
        raise ValueError("输入必须是二维数组")
    # 确保输入形状相同
    if W.shape != W_prime.shape:
        raise ValueError("输入的水印图像必须具有相同的形状")
    # 计算分子部分：W 和 W_prime 的逐点乘积之和
    numerator = np.sum(W * W_prime)
    # 计算分母部分：W 和 W_prime 的平方和的平方根的乘积
    denominator = np.sqrt(np.sum(W * W)) * np.sqrt(np.sum(W_prime * W_prime))
    # 避免除以零的情况
    if denominator == 0:
        return 0.0
    # 计算归一化相关系数
    ncorr = numerator / denominator
    return ncorr


def find_continuous_non_missing(arr, n):
    """
    使用NumPy优化查找
    """
    arr = np.array(arr)
    length = len(arr)
    for i in range(length - n + 1):
        segment = arr[i:i+n]
        if not np.any(segment == -9999):  # 更高效的检查方式
            return segment.tolist()
    return -1


if __name__ == '__main__':
    odg_list =[]
    mse_list = []
    ber_list = []
    corr_list = []
    wm_type = 'DWTDCT'
    # wm_type = "DCTSVD"
    # wm_type = "FFT"
    # wm_type = 'DWTSVD'


    # 同步头-格信息
    _lambda1 = 1
    b1 = _lambda1 * Lattice_Basis('Z', 1)
    N1 = b1.shape[0]
    alpha1 = 2
    coset_representatives1 = cosets(alpha1, N1)
    length1 = round(math.log(len(coset_representatives1), alpha1))

    # 水印-格信息
    _lambda2 = 1
    b2 = _lambda2 * Lattice_Basis('Z', 1)
    N2 = b2.shape[0]
    alpha2 = 2
    coset_representatives2 = cosets(alpha2, N2)
    length2 = round(math.log(len(coset_representatives2), alpha2))

    # 水印信息
    seg_num = 10  # 水印分组数
    W_ori = cv2.imread('watermark.png', cv2.IMREAD_GRAYSCALE)
    W = np.asarray(W_ori, dtype=int)
    W_o = np.copy(W)

    W[W == 255] = 1

    W = W.flatten()
    W_ori_flat = np.copy(W)
    W = Create_Data.trans(W, length2)
    W_seg = [W[i:i + len(W) // seg_num] for i in range(0, len(W), len(W) // seg_num)]
    # 加载音频文件

    audio_path = 'audio_file/disco000.wav'
    signal, sr = librosa.load(audio_path, sr=None)


    # signal = loadmat('Lorenz_data.mat')['data'][0]

    seg_len = 100

    # 生成伪随机序列用于同步头
    # 本原多项式
    L = 9
    polynomial = [1, 0, 0, 0, 0, 1, 0, 0, 0, 1]  # 注意：多项式从高到低表示 L=9
    # 初始状态 [0, 0, 0, 1]
    initial_state = [1, 0, 0, 1, 0, 1, 0, 0, 1]
    # 创建 LFSR
    lfsr = LFSR(polynomial, initial_state)
    # 生成 15 位伪随机序列（最大周期为 2^4 - 1 = 15）
    sequence = lfsr.generate_sequence(2 ** L - 1)

    # 同步头分组 取每20个为一组 共十个同步头
    sync_chunk_size = 30  # 同步头的长度
    all_sync = [sequence[i:i + sync_chunk_size] for i in range(0, len(sequence), sync_chunk_size)][0:seg_num]
    all_sync = np.asarray(all_sync)
    # 将水印嵌入所需的音频分为10组，每组头部使用QIM插入同步头
    wm_signal_len = len(W) * seg_len + sync_chunk_size * seg_num
    wm_signal = np.copy(signal)[0:wm_signal_len]
    extract_sync = np.zeros_like(all_sync)
    # 嵌入同步头和水印
    sync_pos = []
    p = wm_signal_len // seg_num
    W_seg_len = len(W_seg[0])
    for i in range(len(W_seg)):
        for j in range(sync_chunk_size):
            wm_signal[i * (p) + j:i * (p) + j + 1] = standard_qim(wm_signal[i * (p) + j:i * (p) + j + 1], b1,
                                                                  all_sync[i][j], alpha1, coset_representatives1)
        sync_pos.append(i * (p))
        wm_signal[
        i * (p) + sync_chunk_size:i * (
            p) + sync_chunk_size + seg_len * W_seg_len] = new_watermark.qim_watermark_embed(
            wm_signal[i * (p) + sync_chunk_size:i * (p) + sync_chunk_size + seg_len * W_seg_len]
            , W_seg[i], b2, seg_len,wm_type)
    wm_signal2 = np.copy(wm_signal)
    print(sync_pos)
    # sf.write("audio_file/wm_signal.wav", wm_signal, sr)
    # odg_value = calculate_odg(signal[0:wm_signal_len], wm_signal, sr)
    print("水印mse", np.mean((signal[0:wm_signal_len] - wm_signal) ** 2))
    # print("水印ODG:", odg_value)
    print("水印SNR:", calculate_snr(signal[0:wm_signal_len], signal[0:wm_signal_len] - wm_signal))
    mse_list.append(np.mean((signal[0:wm_signal_len] - wm_signal) ** 2))
    # odg_list.append(odg_value)

    # 各种攻击
    # EnKF攻击
    # 保存数据
    wm_data = {'data':wm_signal}
    # savemat('Lorenz_data_wm.mat',wm_data)
    savemat('C:/Users/99409/IdeaProjects/project/remove-watermark-matlab/wav_data_wm.mat', wm_data)

    # GA
    attack_name = "EnKF"
    wm_signal = loadmat("wav_data_EnKF_Q002r75.mat")['data'][0]

    print("攻击后水印mse:", np.mean((signal[0:wm_signal_len] - wm_signal) ** 2))

    extract_W = np.zeros_like(W_seg)
    for i in range(len(W_seg)):
        for j in range(sync_chunk_size):
            extract_sync[i][j] = qim_decode(wm_signal[i * (p) + j:i * (p) + j + 1], b1, alpha1,
                                            coset_representatives1)
        # print(i * (p) + sync_chunk_size, "和", i * (p) + sync_chunk_size + seg_len * W_seg_len)
        extract_W[i] = new_watermark.qim_watermark_extract(
            wm_signal[i * (p) + sync_chunk_size:i * (p) + sync_chunk_size + seg_len * W_seg_len], W_seg[i], b2,
            seg_len,wm_type)
    print(np.array_equal(all_sync, extract_sync))  # 同步头提取正误
    print(np.array_equal(W_seg, extract_W))  # 水印提取正误




    # """ 注意修改m水印corr和提取代码
    # LSC攻击
    # lsc_signal, crop_starts = lsc_attack(wm_signal, 0.2, 10)
    # print("裁剪位置：", crop_starts)
    lsc_signal = wm_signal

    st = time.time()

    # 检测是否有同步头存在
    detected_syncs = []
    # QIM提取所有
    trans_wmsignal = np.copy(lsc_signal)
    num_processes = 20
    # 使用多进程池
    with Pool(processes=num_processes) as pool:
        # 并行解码
        results = pool.starmap(qim_decode, [(signal, b1, alpha1, coset_representatives1) for signal in lsc_signal])
    # 更新解码后的信号
    trans_wmsignal = np.array(results)
    print("从裁剪的音频中解码耗时：", time.time() - st)

    s2 = time.time()
    with Pool(processes=num_processes) as pool:
        # 并行解码
        results = pool.starmap(find_sequence, [(trans_wmsignal, template) for template in all_sync[0:seg_num]])
    pos = np.asarray(results)
    print(pos)
    print("搜索同步头耗时:", time.time() - s2)

    # 生成水印密钥
    # 本原多项式
    L = 12
    # polynomial = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]  # 注意：多项式从高到低表示 L=13
    polynomial = [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]  # L=12
    # 初始状态
    initial_state = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
    # 创建 LFSR
    lfsr = LFSR(polynomial, initial_state)
    # 生成 15 位伪随机序列（最大周期为 2^4 - 1 = 15）
    sequence = lfsr.generate_sequence(2 ** L - 1)

    print("生成的伪随机序列长度:", len(sequence))
    Wkey = W_ori_flat ^ sequence[0:len(W_ori_flat)]

    extracted_watermark = []
    # 原始音频分块 去掉同步头
    original_audio_segments = []
    p = wm_signal_len // seg_num
    for i in range(10):
        # 去掉同步头，得到音频信号分块
        original_audio_segment = signal[i * (p) + sync_chunk_size:i * (p) + sync_chunk_size + seg_len * W_seg_len]
        original_audio_segments.append(original_audio_segment)

    ###############################################################################
    if np.all(pos == -1):
        # m-提取
        original_as = np.concatenate(original_audio_segments)
        extracted_watermark.append(
            wm_extraction.new_qim_m_watermark_extraction(lsc_signal[30:], original_as, Wkey,
                                                         b2,
                                                         W_seg[i]))
        # 恢复二进制水印
        pro_wm = []
        L_bit = 1  # 至少多少bit
        for i in range(len(extracted_watermark)):
            if not np.array_equal(extracted_watermark[i], -1):
                pro_wm.append(find_continuous_non_missing(extracted_watermark[i],L))
        # 恢复部分M序列
        M_seg = []
        for i in range(len(pro_wm)):
            if not np.array_equal(pro_wm[i], -1):
                wkey_seg = np.asarray(Wkey[i * W_seg_len:i * W_seg_len + L])
                temp = np.asarray(pro_wm[i], dtype=int)
                M_seg.append(wkey_seg ^ temp)
            else:
                M_seg.append(-1)

        # 利用部分M序列恢复丢失段（论文Section III-E）
        # 假设存在至少L个连续正确比特
        M_entire_seg = []
        # 周期扩展（循环左移生成完整M序列）
        for i in range(len(M_seg)):
            if not np.array_equal(M_seg[i], -1):
                # 初始状态
                initial_state2 = M_seg[i][::-1].tolist()
                # 创建 LFSR
                lfsr = LFSR(polynomial, initial_state2)
                M_entire_seg.append(lfsr.generate_sequence(2 ** L - 1))
            else:
                M_entire_seg.append(-1)
        # 循环移位计数
        count = np.zeros(len(M_entire_seg))
        for i in range(len(M_entire_seg)):
            for j in range(i, len(M_entire_seg)):
                print(i, "和", j)
                if (not np.array_equal(M_entire_seg[i], -1) and not np.array_equal(M_entire_seg[j],
                                                                                   -1)) and is_cyclic_shift(
                    M_entire_seg[i],
                    M_entire_seg[
                        j]):
                    count[i] += 1
                    count[j] += 1
        # 选出正确的M序列并进行移位产生原始序列
        max_index = np.argmax(count)
        M_ext_sequence = M_entire_seg[max_index]

        # 将移位的M序列分成两部分
        part_num = max_index * 250
        part1 = M_ext_sequence[part_num:]
        part2 = M_ext_sequence[:part_num]
        # 重新组合得到原始M序列
        original_m_sequence = np.concatenate((part1, part2))
        original_m_sequence = np.asarray(original_m_sequence, dtype=int)
        print("提取的M序列是否与原始序列一致：",
              np.array_equal(original_m_sequence, sequence[0:len(original_m_sequence)]))

        restored_watermark = original_m_sequence[0:len(Wkey)] ^ Wkey

        if np.array_equal(W_ori_flat, restored_watermark):
            print("经裁切后，水印可以完整提取")
            corr_list.append(1)
            ber = 0
        else:
            # 计算错误比特数
            error_bits = np.sum(W_ori_flat != restored_watermark)
            # 计算总比特数
            total_bits = W_ori_flat.size
            # 计算误码率
            ber = error_bits / total_bits
            print("BER:", ber)
            print("经裁切后，水印无法完整提取")

    else :
        # 根据返回的索引值提取同步头
        sync_heads = []
        for p in pos:
            if p != -1:
                sync_head = trans_wmsignal[p:p + sync_chunk_size]
                sync_heads.append(sync_head)
            else:
                sync_heads.append(-1)  # 同步头丢失，置为空
        # print("同步头全部取出：", np.array_equal(np.asarray(sync_heads), all_sync))
        p = wm_signal_len // seg_num
        # 根据返回的索引值提取同步头，并去掉同步头
        audio_segments = []
        pos = np.append(pos, -1)
        for i in range(10):
            if pos[i] != -1:
                # 去掉同步头，得到音频信号分块
                audio_segment = lsc_signal[pos[i] + sync_chunk_size:pos[i] + p]
                audio_segments.append(audio_segment)
            else:
                audio_segments.append(-1)

        # 对每个分块进行水印提取
        # m-提取
        for i in range(len(audio_segments)):
            if not np.array_equal(audio_segments[i], -1):
                extracted_watermark.append(
                    wm_extraction.new_qim_m_watermark_extraction(audio_segments[i], original_audio_segments[i], Wkey,
                                                                 b2,
                                                                 W_seg[i]))
            else:
                extracted_watermark.append(-1)
        # 恢复二进制水印
        pro_wm = []
        L_bit = 1  # 至少多少bit
        for i in range(len(extracted_watermark)):
            if not np.array_equal(extracted_watermark[i], -1):
                pro_wm.append(find_continuous_non_missing(extracted_watermark[i],L))
        # 恢复部分M序列
        M_seg = []
        for i in range(len(pro_wm)):
            if not np.array_equal(pro_wm[i], -1):
                wkey_seg = np.asarray(Wkey[i * W_seg_len:i * W_seg_len + L])
                temp = np.asarray(pro_wm[i], dtype=int)
                M_seg.append(wkey_seg ^ temp)
            else:
                M_seg.append(-1)

        # 利用部分M序列恢复丢失段（论文Section III-E）
        # 假设存在至少L个连续正确比特
        M_entire_seg = []
        # 周期扩展（循环左移生成完整M序列）
        for i in range(len(M_seg)):
            if not np.array_equal(M_seg[i], -1):
                # 初始状态
                initial_state2 = M_seg[i][::-1].tolist()
                # 创建 LFSR
                lfsr = LFSR(polynomial, initial_state2)
                M_entire_seg.append(lfsr.generate_sequence(2 ** L - 1))
            else:
                M_entire_seg.append(-1)
        # 循环移位计数
        count = np.zeros(len(M_entire_seg))
        for i in range(len(M_entire_seg)):
            for j in range(i, len(M_entire_seg)):
                print(i, "和", j)
                if (not np.array_equal(M_entire_seg[i], -1) and not np.array_equal(M_entire_seg[j],
                                                                                   -1)) and is_cyclic_shift(
                    M_entire_seg[i],
                    M_entire_seg[
                        j]):
                    count[i] += 1
                    count[j] += 1
        # 选出正确的M序列并进行移位产生原始序列
        max_index = np.argmax(count)
        M_ext_sequence = M_entire_seg[max_index]

        # 将移位的M序列分成两部分
        part_num = max_index * 250
        part1 = M_ext_sequence[part_num:]
        part2 = M_ext_sequence[:part_num]
        # 重新组合得到原始M序列
        original_m_sequence = np.concatenate((part1, part2))
        original_m_sequence = np.asarray(original_m_sequence, dtype=int)
        print("提取的M序列是否与原始序列一致：",
              np.array_equal(original_m_sequence, sequence[0:len(original_m_sequence)]))

        restored_watermark = original_m_sequence[0:len(Wkey)] ^ Wkey

        if np.array_equal(W_ori_flat, restored_watermark):
            print("经裁切后，水印可以完整提取")
            ber = 0
        else:
            # 计算错误比特数
            error_bits = np.sum(W_ori_flat != restored_watermark)
            # 计算总比特数
            total_bits = W_ori_flat.size
            # 计算误码率
            ber = error_bits / total_bits
            print("BER:", ber)
            print("经裁切后，水印无法完整提取")
    data_reshaped = np.asarray(restored_watermark)
    matrix = data_reshaped.reshape(50, 50)
    matrix[matrix == 1] = 255
    ber_list.append(ber)
    corr_list.append(calculate_ncorr(W_o, matrix))

    # 将NumPy数组转换为图像
    matrix2 = matrix.astype(np.uint8)
    attack_name = "EnKF_r75"
    attack_name = attack_name + '_image.png'
    cv2.imwrite(attack_name, matrix2)

    # 计算错误比特数
    error_bits = np.sum(matrix2 != W_o)
    # 计算总比特数
    total_bits = matrix2.size
    # 计算误码率
    ber = error_bits / total_bits
    print("水印BER:", ber)
    print("水印corr:", calculate_ncorr(W_o, matrix))



# """
#     print("odg",odg_list)
#     print(sum(odg_list) / len(odg_list))

    print("mse", mse_list)
    print(sum(mse_list) /len(mse_list))

    print("ber", ber_list)
    print(sum(ber_list) / len(ber_list))

    print("ncorr", corr_list)
    print(sum(corr_list) / len(corr_list))


