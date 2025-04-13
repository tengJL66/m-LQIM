import pywt
import numpy as np
import math
from lattice_embed.QIM import standard_qim, cosets, qim_decode
from lattice_embed.Lattice.Lattice_basis import Lattice_Basis
from lattice_embed import Create_Data


# 图邻接矩阵
def generate_adjacency_matrix(n):
    # 创建一个n*n的矩阵，初始值为0
    matrix = np.zeros((n, n))

    # 填充相邻节点的权重为1
    for i in range(n - 1):
        matrix[i, (i + 1) % n] = 1
        matrix[(i + 1) % n, i] = 1

    # 填充相隔一个节点的权重为0.1
    for i in range(n - 1):
        matrix[i, (i + 2) % n] = 0.1
        matrix[(i + 2) % n, i] = 0.1

    return matrix


# 计算稀疏系数
def DGS_parameter(signal):
    wavelet = 'db1'
    coeffs = pywt.wavedec(signal, wavelet, level=3)
    cA3, cD3, cD2, cD1 = coeffs

    # --- GBT变换修正 ---
    n = len(cA3)
    A = generate_adjacency_matrix(n)
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    K = D - A
    vals, vecs = np.linalg.eig(K)

    # 对特征值和特征向量排序
    idx = np.argsort(vals)  # 按特征值升序
    vecs = vecs[:, idx]  # 调整特征向量顺序

    # 使用转置代替逆矩阵
    G = cA3 @ vecs.T  # 修正点1

    # --- SVD分解与嵌入修正 ---
    # U, C, VT = np.linalg.svd(np.diag(G))
    U, C, VT = SVD(np.diag(G))
    return C


# def DGS_parameter(signal):
#     # 三级DWT变换
#     # 选择小波基函数
#     wavelet = 'db1'  # Daubechies小波
#     # 进行三级离散小波变换
#     coeffs = pywt.wavedec(signal, wavelet, level=3)
#     cA3, cD3, cD2, cD1 = coeffs  # 分别对应第3级近似系数和第3、2、1级细节系数
#
#     # 生成GBT所需正交矩阵
#     n = len(cA3)  # n为近似系数的长度
#     A = generate_adjacency_matrix(n)  # 邻接矩阵A
#     # 计算每个节点的度数
#     degrees = np.sum(A, axis=1)
#
#     # 构造度矩阵 D
#     D = np.diag(degrees)
#     K = D - A
#
#     # 使用 np.linalg.eig 进行特征值分解
#     vals, vecs = np.linalg.eig(K)
#     # GBT变换
#     # G = cA3 @ np.linalg.inv(vecs)
#     G = cA3 @ vecs.T
#     Gd = np.diag(G)
#     # SVD分解
#     U, C, VT = np.linalg.svd(Gd)
#     return C


def DG_parameter(signal):
    # 三级DWT变换
    # 选择小波基函数
    wavelet = 'db1'
    coeffs = pywt.wavedec(signal, wavelet, level=3)
    cA3, cD3, cD2, cD1 = coeffs

    # --- GBT变换修正 ---
    n = len(cA3)
    A = generate_adjacency_matrix(n)
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    K = D - A
    vals, vecs = np.linalg.eig(K)

    # 对特征值和特征向量排序
    idx = np.argsort(vals)  # 按特征值升序
    vecs = vecs[:, idx]  # 调整特征向量顺序

    # 使用转置代替逆矩阵
    G = cA3 @ vecs.T  # 修正点1
    return G


#
# def qim_watermark_embed(signal, W,b):
#     wm_signal = np.copy(signal)
#     # QIM参数
#     N = b.shape[0]
#     alpha = 2
#     coset_representatives = cosets(alpha, N)
#     length = round(math.log(len(coset_representatives), alpha))
#     # create secret messages 生成
#     W = Create_Data.trans(W, length)
#
#
#
#     for i in range(len(W)):
#         if i==12:
#             print()
#         signal_segment = signal[i * 100:(i + 1) * 100]
#         # 选择小波基函数
#         wavelet = 'db1'  # Daubechies小波
#         # 进行三级离散小波变换
#         coeffs = pywt.wavedec(signal_segment, wavelet, level=3)
#         cA3, cD3, cD2, cD1 = coeffs  # 分别对应第3级近似系数和第3、2、1级细节系数
#
#         # 生成GBT所需正交矩阵
#         n = len(cA3)  # n为近似系数的长度
#         A = generate_adjacency_matrix(n)  # 邻接矩阵A
#
#         # 计算每个节点的度数
#         degrees = np.sum(A, axis=1)
#
#         # 构造度矩阵 D
#         D = np.diag(degrees)
#         K = D - A
#
#         # 使用 np.linalg.eig 进行特征值分解
#         # 在特征分解后添加排序逻辑
#         vals, vecs = np.linalg.eig(K)
#
#         # 按特征值从小到大排序
#         idx = np.argsort(vals)
#         vals = vals[idx]
#         vecs = vecs[:, idx]  # 按列排序
#
#         # GBT变换
#         G = cA3 @ vecs.T
#         Gd = np.diag(G)
#         # SVD分解
#         U, C, VT = np.linalg.svd(Gd)
#         C_embed = np.copy(C)
#
#         # QIM 水印嵌入
#         C_embed[0:N] = standard_qim(C_embed[0:N], b, W[i], alpha, coset_representatives)
#
#         print(W[i], "和", qim_decode(C_embed[0:N], b, alpha, coset_representatives))
#         # SVD还原
#         C_embed_diag = np.diag(C_embed)
#         G_embed = U @ C_embed_diag @ VT
#         G_embed = np.diag(G_embed)
#         # IGBT还原
#         D_embed = G_embed @ vecs
#
#         coeffs_new = [D_embed, cD3, cD2, cD1]
#         # IDWT还原
#         wm_signal_segment = pywt.waverec(coeffs_new, wavelet)
#         wm_signal[i * 100:(i + 1) * 100] = wm_signal_segment
#
#         C_s = DGS_parameter(wm_signal_segment)
#         if qim_decode(C_s[0:N], b, alpha, coset_representatives)!=W[i]:
#             print("一系列操作后：",qim_decode(C_s[0:N], b, alpha, coset_representatives))
#
#
#     return wm_signal




def SVD(A):
    # 计算 A^T A 和 A A^T
    A_T_A = np.dot(A.T, A)
    A_A_T = np.dot(A, A.T)

    # 计算 A^T A 的特征值和特征向量
    eigenvalues_V, V = np.linalg.eig(A_T_A)
    # eigenvalues_V = np.abs(eigenvalues_V)  # 取绝对值

    # 计算 A A^T 的特征值和特征向量
    eigenvalues_U, U = np.linalg.eig(A_A_T)
    # eigenvalues_U = np.abs(eigenvalues_U)  # 取绝对值

    # 提取奇异值
    singular_values = np.sqrt(eigenvalues_V)

    # 保持符号
    sign_matrix = np.sign(A)
    V = V @ sign_matrix

    return U, singular_values, V.T



def qim_watermark_embed(signal, W, b, _lambda):
    wm_signal = np.copy(signal)
    N = b.shape[0]
    alpha = 2
    coset_representatives = cosets(alpha, N)
    length = round(math.log(len(coset_representatives), alpha))
    W = Create_Data.trans(W, length)
    j = 0  # 音频信号索引
    i = 0  # 水印索引
    while (i < len(W)):
        if j >= len(signal) - 100:
            break
        if i == 1045:
            print()
        signal_segment = signal[j:j + 100]
        wavelet = 'db1'
        coeffs = pywt.wavedec(signal_segment, wavelet, level=3)
        cA3, cD3, cD2, cD1 = coeffs

        # --- GBT变换修正 ---
        n = len(cA3)
        A = generate_adjacency_matrix(n)
        degrees = np.sum(A, axis=1)
        D = np.diag(degrees)
        K = D - A
        vals, vecs = np.linalg.eig(K)

        # 对特征值和特征向量排序
        idx = np.argsort(vals)  # 按特征值升序
        vecs = vecs[:, idx]  # 调整特征向量顺序

        # 使用转置代替逆矩阵
        G = cA3 @ vecs.T  # 修正点1

        # --- SVD分解与嵌入修正 ---
        # U, C, VT = np.linalg.svd(np.diag(G))
        U, C, VT = SVD(np.diag(G))
        C_embed = np.copy(C)

        C_embed[0:N] = standard_qim(C_embed[0:N], b, W[i], alpha, coset_representatives)

        # --- 逆SVD重构 ---
        C_embed_diag = np.diag(C_embed)
        G_embed = U @ C_embed_diag @ VT
        G_embed = np.diag(G_embed)
        # --- 逆GBT修正 ---
        D_embed = G_embed @ vecs  # 使用排序后的vecs

        coeffs_new = [D_embed, cD3, cD2, cD1]
        wm_signal_segment = pywt.waverec(coeffs_new, wavelet)
        wm_signal[j:j + 100] = wm_signal_segment

        C_s = DGS_parameter(wm_signal_segment)
        if qim_decode(C_s[0:N], b, alpha, coset_representatives) != W[i]:
            print(W[i], "和", qim_decode(C_embed[0:N], b, alpha, coset_representatives))
            print("一系列操作后：", qim_decode(C_s[0:N], b, alpha, coset_representatives))
        i += 1
        j += 100
    return wm_signal


def qim_watermark_extract(wm_signal, W, b, _lambda):
    # QIM参数
    N = b.shape[0]
    alpha = 2
    coset_representatives = cosets(alpha, N)
    length = round(math.log(len(coset_representatives), alpha))
    # create secret messages 生成
    W = Create_Data.trans(W, length)
    w_extract = np.zeros_like(W)
    i = 0
    j = 0
    while (i < len(W)):
        if j ==len(wm_signal):
            break
        s = wm_signal[j:j + 100]
        C_s = DGS_parameter(s)
        w_extract[i] = qim_decode(C_s[0:N], b, alpha, coset_representatives)
        i += 1
        j += 100
    if np.array_equal(w_extract, W):
        print("嵌入后直接提取，结果correct")
    else:
        # 计算错误比特数
        error_bits = np.sum(w_extract != W)
        # 计算总比特数
        total_bits = w_extract.size
        # 计算误码率
        ber = error_bits / total_bits
        print("BER:", ber)
        print("嵌入后直接提取，水印无法完整提取")
    return w_extract
