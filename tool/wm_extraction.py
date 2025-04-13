import numpy as np
import pywt

from lattice_embed import Create_Data
from lattice_embed.QIM import cosets, qim_decode
from tool import qim_watermark
from tool import ss_watermark
from tool.LFSR import LFSR
import math
from scipy.fft import dct,idct
from tool.new_watermark import SVD
def new_qim_m_watermark_extraction(
        attacked_audio,  # 受攻击的音频信号（一维数组）
        original_audio,  # 原始音频分段列表（每个元素为一维数组）
        Wkey,  # 水印密钥（与M序列异或后的结果）
        b,
        W,
        L=13,  # M序列的级数（论文默认L=13）
        thr1=0.5,  # 相似度阈值1（控制步长切换）
        thr2=0.7,  # 相似度阈值2（判断有效分段）
        t1=5,  # 小步长（精细搜索）
        t2=50,  # 大步长（快速跳过）
):
    """完整的水印提取与恢复实现"""
    # QIM参数
    N = b.shape[0]
    alpha = 2
    coset_representatives = cosets(alpha, N)

    # 将原信号进行分割
    # 分割成长度为100的子列表
    chunk_size = 100
    original_audio_segments = [original_audio[i:i + chunk_size] for i in range(0, len(original_audio), chunk_size)]

    restored_watermark = np.full(len(W),-9999,dtype=int)
    best_loc = 0
    for i in range(len(W)):

        Ai = original_audio_segments[i]
        len_Ai = len(Ai)
        max_corr = -np.inf
        Dis = 0  # 总滑动距离（关键参数）

        # 初始位置逻辑（论文Section III-D步骤2）
        if i == 0:
            Loc = 0
        else:
            # 若前一段未检测到，回退到上一段的起始位置
            if np.all(restored_watermark[i - 1] == -9999):
                Loc = Loc - Dis
            else:
                Loc = best_loc + len_Ai

        wavelet = 'db1'  # Daubechies小波
        # CA = dct(Ai, 2, norm="ortho")
        coeffs = pywt.wavedec(Ai, wavelet, level=3)
        CA, cD3, cD2, cD1= coeffs
        # 滑动窗口搜索（论文Section III-D步骤3-6）
        while Loc <= len(attacked_audio) - len_Ai:
            S = attacked_audio[Loc:Loc + len_Ai]
            # 计算原始分段Ai和滑动窗口S的DWT-GBT-SVD系数
            # CS = dct(S, 2, norm="ortho")

            # dwt+dct
            # 进行三级离散小波变换
            coeffs = pywt.wavedec(S, wavelet, level=3)
            CS, cD3, cD2, cD1 = coeffs  # 分别对应第3级近似系数和第3、2、1级细节系数
            # 计算相关系数Corr（皮尔逊）
            corr = np.corrcoef(CA, CS)[0, 1]

            # 更新最大相似度记录
            if corr > max_corr:
                max_corr = corr
                best_loc = Loc

            # 动态调整步长（论文Section III-D步骤3）
            if corr >= thr1:
                stp = t1  # 高相似度区域，精细搜索
            else:
                stp = t2  # 低相似度区域，快速跳过

            # 更新滑动距离和位置（论文中的Dis参数）
            Dis += stp
            Loc += stp

            # 终止条件：滑动距离超过分段长度（论文Section III-D步骤5）
            if Dis >= len_Ai:
                break

        # 判断是否找到有效分段（嵌入水印位置）
        if max_corr >= thr2:
            # #QIM 水印
            S_best = attacked_audio[best_loc:best_loc + len_Ai]
            # CS = dct(S_best, 2, norm="ortho")
            # U,SS,VT = SVD(np.diag(CS))
            # restored_watermark[i] = qim_decode(SS[20:20+N], b, alpha, coset_representatives)

            #dwtdctsvd 水印提取
            wavelet = 'db1'  # Daubechies小波
            # 进行三级离散小波变换
            coeffs = pywt.wavedec(S_best, wavelet, level=3)
            CS, cD3, cD2, cD1 = coeffs  # 分别对应第3级近似系数和第3、2、1级细节系数
            D = dct(CS, type=2, norm="ortho")
            U, S, VT = SVD(np.diag(D))
            restored_watermark[i] = qim_decode(S[1:1 + N], b, alpha, coset_representatives)
        else:
            restored_watermark[i] = -9999 # 标记为被裁剪段
        # 记录前一段的位置和滑动距离（用于下一段初始化）
        Loc = best_loc
        # print(best_loc)
    com = np.where(W==restored_watermark,1,-1)
    # all_restored_watermark = np.zeros_like(Wkey)
    # # 利用M序列恢复丢失段（论文Section III-E）
    # # 假设存在至少L个连续正确比特
    # for i in range(len(restored_watermark)):
    #     if restored_watermark[i]!= -9999:
    #         all_restored_watermark[N*i:N*i+N] = Create_Data.trans_back(restored_watermark[i])
    #     else:
    #         all_restored_watermark[N * i:N * i + N] = np.array([-9999 ]* N)
    #
    #
    # M = np.full(Wkey.shape,-9999,dtype=int)
    # for i in range(len(Wkey)):
    #     if np.any(all_restored_watermark[i] != -9999):
    #         M[i] = all_restored_watermark[i] ^ Wkey[i]
    #
    #
    # # 周期扩展（循环左移生成完整M序列）
    # polynomial = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]  # 注意：多项式从高到低表示 L=13
    # # 初始状态
    # initial_state = M[0:L].tolist()[::-1]
    # # 创建 LFSR
    # lfsr = LFSR(polynomial, initial_state)
    # M_full = lfsr.generate_sequence(2**L-1)
    #
    # restored_watermark = M_full[0:len(Wkey)] ^ Wkey

    # 混沌解密（需实现逆混沌映射）
    # final_watermark = chaotic_decrypt(restored_watermark)
    # final_watermark = 0
    return restored_watermark

def qim_m_watermark_extraction(
        attacked_audio,  # 受攻击的音频信号（一维数组）
        original_audio,  # 原始音频分段列表（每个元素为一维数组）
        Wkey,  # 水印密钥（与M序列异或后的结果）
        _lambda,  # 嵌入强度因子
        b,
        L=13,  # M序列的级数（论文默认L=13）
        thr1=0.5,  # 相似度阈值1（控制步长切换）
        thr2=0.8,  # 相似度阈值2（判断有效分段）
        t1=5,  # 小步长（精细搜索）
        t2=50,  # 大步长（快速跳过）
):
    """完整的水印提取与恢复实现"""
    # QIM参数
    N = b.shape[0]
    alpha = 2
    coset_representatives = cosets(alpha, N)
    length = round(math.log(len(coset_representatives), alpha))
    # create secret messages 生成
    W = Create_Data.trans(Wkey, length)


    # 将原信号进行分割
    # 分割成长度为100的子列表
    chunk_size = 100
    original_audio_segments = [original_audio[i:i + chunk_size] for i in range(0, len(original_audio), chunk_size)]

    restored_watermark = np.full(len(W),-9999,dtype=int)
    best_loc = 0
    for i in range(len(W)):

        Ai = original_audio_segments[i]
        len_Ai = len(Ai)
        max_corr = -np.inf
        Dis = 0  # 总滑动距离（关键参数）

        # 初始位置逻辑（论文Section III-D步骤2）
        if i == 0:
            Loc = 0
        else:
            # 若前一段未检测到，回退到上一段的起始位置
            if np.all(restored_watermark[i - 1] == -9999):
                Loc = Loc - Dis
            else:
                Loc = best_loc + len_Ai

        # 滑动窗口搜索（论文Section III-D步骤3-6）
        while Loc <= len(attacked_audio) - len_Ai:
            S = attacked_audio[Loc:Loc + len_Ai]

            # 计算原始分段Ai和滑动窗口S的DWT-GBT-SVD系数
            Gi_Ai = qim_watermark.DG_parameter(Ai)
            Gi_S = qim_watermark.DG_parameter(S)

            # 计算相关系数Corr（皮尔逊）
            corr = np.corrcoef(Gi_Ai, Gi_S)[0, 1]

            # 更新最大相似度记录
            if corr > max_corr:
                max_corr = corr
                best_loc = Loc

            # 动态调整步长（论文Section III-D步骤3）
            if corr >= thr1:
                stp = t1  # 高相似度区域，精细搜索
            else:
                stp = t2  # 低相似度区域，快速跳过

            # 更新滑动距离和位置（论文中的Dis参数）
            Dis += stp
            Loc += stp

            # 终止条件：滑动距离超过分段长度（论文Section III-D步骤5）
            if Dis >= len_Ai:
                break

        # 判断是否找到有效分段（嵌入水印位置）
        if max_corr >= thr2:
            # # ss水印
            # S_best = attacked_audio[best_loc:best_loc + len_Ai]
            # Ci_attacked = DGS_parameter(S_best)
            # # 从Gi_attacked中提取水印段（需保存原始Ci）
            # Ci_original = DGS_parameter(Ai)  # 假设已预先存储
            # # 水印提取 嵌入在哪个位置？
            # Wi = (Ci_attacked[0] - Ci_original[0]) / _lambda
            # restored_watermark[i] = round(Wi)

            #QIM 水印
            S_best = attacked_audio[best_loc:best_loc + len_Ai]
            Ci_attacked = qim_watermark.DGS_parameter(S_best)
            restored_watermark[i] = qim_decode(Ci_attacked[0:N], b, alpha, coset_representatives)
        else:
            restored_watermark[i] = -9999 # 标记为被裁剪段

        # 记录前一段的位置和滑动距离（用于下一段初始化）
        Loc = best_loc

    all_restored_watermark = np.zeros_like(Wkey)
    # 利用M序列恢复丢失段（论文Section III-E）
    # 假设存在至少L个连续正确比特
    for i in range(len(restored_watermark)):
        if restored_watermark[i]!= -9999:
            all_restored_watermark[N*i:N*i+N] = Create_Data.trans_back(restored_watermark[i])
        else:
            all_restored_watermark[N * i:N * i + N] = np.array([-9999 ]* N)

    M = np.full(Wkey.shape,-9999,dtype=int)
    for i in range(len(Wkey)):
        if np.any(all_restored_watermark[i] != -9999):
            M[i] = all_restored_watermark[i] ^ Wkey[i]

    # 周期扩展（循环左移生成完整M序列）
    polynomial = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]  # 注意：多项式从高到低表示 L=13
    # 初始状态
    initial_state = M[0:L].tolist()[::-1]
    # 创建 LFSR
    lfsr = LFSR(polynomial, initial_state)
    M_full = lfsr.generate_sequence(2**L-1)

    restored_watermark = M_full[0:len(Wkey)] ^ Wkey

    # 混沌解密（需实现逆混沌映射）
    # final_watermark = chaotic_decrypt(restored_watermark)
    # final_watermark = 0
    return restored_watermark


def ss_m_watermark_extraction(
        attacked_audio,  # 受攻击的音频信号（一维数组）
        original_audio,  # 原始音频分段列表（每个元素为一维数组）
        Wkey,  # 水印密钥（与M序列异或后的结果）
        _lambda,  # 嵌入强度因子
        M_ori,
        L=13,  # M序列的级数（论文默认L=13）
        thr1=0.5,  # 相似度阈值1（控制步长切换）
        thr2=0.8,  # 相似度阈值2（判断有效分段）
        t1=5,  # 小步长（精细搜索）
        t2=50,  # 大步长（快速跳过）
):
    """完整的水印提取与恢复实现"""
    # 将原信号进行分割
    # 分割成长度为100的子列表
    chunk_size = 100
    original_audio_segments = [original_audio[i:i + chunk_size] for i in range(0, len(original_audio), chunk_size)]

    restored_watermark = np.full(len(Wkey),-9999,dtype=int)
    best_loc = 0
    for i in range(len(Wkey)):

        Ai = original_audio_segments[i]
        len_Ai = len(Ai)
        max_corr = -np.inf
        Dis = 0  # 总滑动距离（关键参数）

        # 初始位置逻辑（论文Section III-D步骤2）
        if i == 0:
            Loc = 0
        else:
            # 若前一段未检测到，回退到上一段的起始位置
            if np.all(restored_watermark[i - 1] == -9999):
                Loc = Loc - Dis
            else:
                Loc = best_loc + len_Ai

        # 滑动窗口搜索（论文Section III-D步骤3-6）
        while Loc <= len(attacked_audio) - len_Ai:
            S = attacked_audio[Loc:Loc + len_Ai]

            # 计算原始分段Ai和滑动窗口S的DWT-GBT-SVD系数
            Gi_Ai = ss_watermark.DG_parameter(Ai)
            Gi_S = ss_watermark.DG_parameter(S)

            # 计算相关系数Corr（皮尔逊）
            corr = np.corrcoef(Gi_Ai, Gi_S)[0, 1]

            # 更新最大相似度记录
            if corr > max_corr:
                max_corr = corr
                best_loc = Loc

            # 动态调整步长（论文Section III-D步骤3）
            if corr >= thr1:
                stp = t1  # 高相似度区域，精细搜索
            else:
                stp = t2  # 低相似度区域，快速跳过

            # 更新滑动距离和位置（论文中的Dis参数）
            Dis += stp
            Loc += stp

            # 终止条件：滑动距离超过分段长度（论文Section III-D步骤5）
            if Dis >= len_Ai:
                break

        # 判断是否找到有效分段（嵌入水印位置）
        if max_corr >= thr2:
            # # ss水印
            S_best = attacked_audio[best_loc:best_loc + len_Ai]
            Ci_attacked = ss_watermark.DGS_parameter(S_best)
            # 从Gi_attacked中提取水印段（需保存原始Ci）
            Ci_original = ss_watermark.DGS_parameter(Ai)  # 假设已预先存储
            # 水印提取 嵌入在哪个位置？
            Wi = (Ci_attacked[0] - Ci_original[0]) / _lambda
            restored_watermark[i] = round(Wi)
        else:
            restored_watermark[i] = -9999 # 标记为被裁剪段

        # 记录前一段的位置和滑动距离（用于下一段初始化）
        Loc = best_loc

    # 利用M序列恢复丢失段（论文Section III-E）
    # 假设存在至少L个连续正确比特

    M = np.full(Wkey.shape,-9999,dtype=int)
    for i in range(len(Wkey)):
        if np.any(restored_watermark[i] != -9999):
            M[i] = restored_watermark[i] ^ Wkey[i]

    comparison = np.where(M == M_ori,1,-1)
    # 周期扩展（循环左移生成完整M序列）
    polynomial = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]  # 注意：多项式从高到低表示 L=13
    # 初始状态
    initial_state = M[0:L].tolist()[::-1]
    # 创建 LFSR
    lfsr = LFSR(polynomial, initial_state)
    M_full = lfsr.generate_sequence(2**L-1)

    restored_watermark = M_full[0:len(Wkey)] ^ Wkey



    # 混沌解密（需实现逆混沌映射）
    # final_watermark = chaotic_decrypt(restored_watermark)
    # final_watermark = 0
    return restored_watermark


