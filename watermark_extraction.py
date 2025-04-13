import numpy as np
from tool.ss_watermark import *


def compute_correlation(x, y):
    """
    计算两个一维信号的皮尔逊相关系数。
    参数:
        x (np.ndarray): 第一个信号，形状为 (N,)。
        y (np.ndarray): 第二个信号，形状为 (N,)。

    返回:
        float: 皮尔逊相关系数，范围 [-1, 1]。
    """
    # 计算均值
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # 计算协方差
    covariance = np.sum((x - mean_x) * (y - mean_y))

    # 计算标准差
    std_x = np.sqrt(np.sum((x - mean_x) ** 2))
    std_y = np.sqrt(np.sum((y - mean_y) ** 2))

    # 计算相关系数
    if std_x == 0 or std_y == 0:
        return 0  # 避免除零错误
    else:
        return covariance / (std_x * std_y)


def watermark_extraction(attacked_audio, original_audio, Wkey, segment_len, thr1, thr2, t1, t2):
    n_segments = segment_len
    restored_watermark = np.zeros_like(Wkey)

    for i in range(np.floor(original_audio / segment_len)):
        Ai = original_audio[i:i + segment_len]
        max_corr = 0
        best_loc = 0
        if i == 0:
            loc = 0
        else :
            loc = 0


        while loc <= len(attacked_audio) - len(Ai):
            S = attacked_audio[loc:loc + len(Ai)]
            # 计算DWT-GBT-SVD系数
            Gi, Gs = DGS_parameter(Ai), DGS_parameter(S)
            corr = compute_correlation(Gi, Gs)

            if corr >= thr1:
                stp = t1  # 精细搜索
            else:
                stp = t2  # 快速跳过

            if corr > max_corr:
                max_corr = corr
                best_loc = loc

            loc += stp

        if max_corr >= thr2:
            # 提取水印段
            S_best = attacked_audio[best_loc:best_loc + len(Ai)]
            S, Gi_attacked, VT = DGS_all_parameter(S_best)
            Ci_diag = np.diag(Gi_attacked)
            Ci_attacked = U @ Ci_diag @ VT
            Wi = (Ci_attacked - original_audio.Ci[i]) / lambda_
            restored_watermark[i] = Wi
        else:
            restored_watermark[i] = 0  # 标记为裁剪段

    # # 利用M序列恢复丢失段
    # M = recover_m_sequence(restored_watermark, Wkey, L)
    # for j in range(n_segments):
    #     if np.all(restored_watermark[j] == 0):
    #         Mj = generate_m_segment(M, j)
    #         restored_watermark[j] = Mj ^ Wkey[j]
    #
    # # 混沌解密
    # final_watermark = chaotic_decrypt(restored_watermark)
    return final_watermark


if __name__ == '__main__':
    print(np.floor(150 / 4))
