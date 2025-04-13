import numpy as np
from scipy.signal import find_peaks
import pywt

def generate_m_sequence(order):
    """
    生成M序列
    :param order: 序列阶数
    :return: M序列
    """
    # 初始化寄存器
    register = np.ones(order, dtype=int)
    # 生成M序列
    sequence = []
    for _ in range(2**order - 1):  # M序列的周期为2^order - 1
        feedback = register[0] ^ register[order - 1]  # 反馈函数
        sequence.append(feedback)
        register = np.roll(register, -1)  # 寄存器右移
        register[-1] = feedback
    return np.array(sequence)

# 示例：生成10阶M序列
m_sequence = generate_m_sequence(10)
print("M序列：", m_sequence)


def embed_sync_header(audio_segment, sync_header):
    """
    将同步头嵌入到音频分块的DWT低频子带中
    :param audio_segment: 音频分块
    :param sync_header: 同步头
    :return: 嵌入同步头后的音频分块
    """
    # 对音频分块进行3层DWT分解
    coeffs = pywt.wavedec(audio_segment, 'db1', level=3)
    # 提取低频子带（LL3）
    ll3 = coeffs[0]
    # 量化嵌入同步头
    quantized_ll3 = ll3 + sync_header
    # 更新低频子带
    coeffs[0] = quantized_ll3
    # 重构音频分块
    embedded_segment = pywt.waverec(coeffs, 'db1')
    return embedded_segment

# 示例：嵌入同步头
audio_segment = np.random.rand(1024)  # 假设音频分块长度为1024
embedded_segment = embed_sync_header(audio_segment, m_sequence[:len(audio_segment)//8])
print("嵌入同步头后的音频分块：", embedded_segment)

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


def load_sync_template(sync_id):
    """
    加载预存的同步头模板
    :param sync_id: 同步头编号
    :return: 同步头模板
    """
    # 示例：加载预存的同步头模板
    sync_templates = {
        1: generate_m_sequence(10),
        2: generate_m_sequence(10),
        # 更多同步头模板...
    }
    return sync_templates[sync_id]


if __name__ == '__main__':

    # 示例：检测同步头
    threshold = 0.9  # 检测阈值
    detected_peaks = detect_sync(embedded_segment, m_sequence[:len(audio_segment) // 8], threshold)
    print("检测到的同步头位置：", detected_peaks)

    # 对裁剪后的音频遍历检测所有可能同步头
    detected_syncs = []
    A2 = np.random.rand(5000)  # 假设裁剪后的音频长度为5000
    for sync_id in range(1, 11):  # 假设有10个同步头模板
        template = load_sync_template(sync_id)  # 加载预存的同步头模板
        peaks = detect_sync(A2, template[:len(A2)//8], threshold)
        detected_syncs.extend([(pos, sync_id) for pos in peaks])

    # 按位置排序并去重，得到有效分块位置
    detected_syncs.sort(key=lambda x: x[0])
    unique_positions = list(set([pos for pos, _ in detected_syncs]))
    print("检测到的同步头位置：", unique_positions)