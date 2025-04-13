import pywt
import numpy as np


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
    # 三级DWT变换
    # 选择小波基函数
    wavelet = 'db1'  # Daubechies小波
    # 进行三级离散小波变换
    coeffs = pywt.wavedec(signal, wavelet, level=3)
    cA3, cD3, cD2, cD1 = coeffs  # 分别对应第3级近似系数和第3、2、1级细节系数

    # 生成GBT所需正交矩阵
    n = len(cA3)  # n为近似系数的长度
    A = generate_adjacency_matrix(n)  # 邻接矩阵A
    # 计算每个节点的度数
    degrees = np.sum(A, axis=1)

    # 构造度矩阵 D
    D = np.diag(degrees)
    K = D - A

    # 使用 np.linalg.eig 进行特征值分解
    vals, vecs = np.linalg.eig(K)
    # GBT变换
    G = cA3 @ np.linalg.inv(vecs)
    Gd = np.diag(G)
    # SVD分解
    U, C, VT = np.linalg.svd(Gd)
    return C
# GS
def GS_parameter(signal):

    # 生成GBT所需正交矩阵
    A = generate_adjacency_matrix(len(signal))  # 邻接矩阵A
    # 计算每个节点的度数
    degrees = np.sum(A, axis=1)
    # 构造度矩阵 D
    D = np.diag(degrees)
    K = D - A
    # 使用 np.linalg.eig 进行特征值分解
    vals, vecs = np.linalg.eig(K)
    # GBT变换
    G = signal @ np.linalg.inv(vecs)
    Gd = np.diag(G)
    # SVD分解
    U, C, VT = np.linalg.svd(Gd)

    return C

def DG_parameter(signal):
    # 三级DWT变换
    # 选择小波基函数
    wavelet = 'db1'  # Daubechies小波
    # 进行三级离散小波变换
    coeffs = pywt.wavedec(signal, wavelet, level=3)
    cA3, cD3, cD2, cD1 = coeffs  # 分别对应第3级近似系数和第3、2、1级细节系数

    # 生成GBT所需正交矩阵
    n = len(cA3)  # n为近似系数的长度
    A = generate_adjacency_matrix(n)  # 邻接矩阵A
    # 计算每个节点的度数
    degrees = np.sum(A, axis=1)

    # 构造度矩阵 D
    D = np.diag(degrees)
    K = D - A

    # 使用 np.linalg.eig 进行特征值分解
    vals, vecs = np.linalg.eig(K)
    # GBT变换
    G = cA3 @ vecs.T
    return G


def GS_watermark_embed(signal, W, _lambda):
    wm_signal = np.copy(signal)

    for i in range(len(W)):
        signal_segment = signal[i * 100:(i + 1) * 100]

        A = generate_adjacency_matrix(len(signal_segment))  # 邻接矩阵A
        # 计算每个节点的度数
        degrees = np.sum(A, axis=1)
        # 构造度矩阵 D
        D = np.diag(degrees)
        K = D - A
        # 使用 np.linalg.eig 进行特征值分解
        vals, vecs = np.linalg.eig(K)
        # GBT变换
        G = signal_segment @ np.linalg.inv(vecs)
        Gd = np.diag(G)
        # SVD分解
        U, C, VT = np.linalg.svd(Gd)
        C_embed = np.copy(C)

        # 水印嵌入
        C_embed[0] = C[0] + _lambda * W[i]

        # SVD还原
        C_embed_diag = np.diag(C_embed)
        G_embed = U @ C_embed_diag @ VT
        G_embed = np.diag(G_embed)
        # IGBT还原
        D_embed = G_embed @ vecs
        wm_signal[i * 100:(i + 1) * 100] = D_embed
    return wm_signal

def GS_watermark_extract(wm_signal, signal,W,_lambda):
    w_extract = np.zeros(W.shape)

    for i in range(len(W)):
        s = wm_signal[i * 100:(i + 1) * 100]
        C_s = GS_parameter(s)
        C_ori = GS_parameter(signal[i * 100:(i + 1) * 100])

        w_extract[i] = (C_s[0] - C_ori[0]) / _lambda
    # 像素 四舍五入
    w_extract = np.round(w_extract)
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

def ss_watermark_embed(signal, W, _lambda):
    wm_signal = np.copy(signal)

    signal_seg1 = signal[0:0 + 100]
    A1 = DGS_parameter(signal_seg1)
    for i in range(200):
        signal_seg = signal[i :i+100]
        A2 = DGS_parameter(signal_seg)


    for i in range(len(W)):
        signal_segment = signal[i * 100:(i + 1) * 100]
        # 选择小波基函数
        wavelet = 'db1'  # Daubechies小波
        # 进行三级离散小波变换
        coeffs = pywt.wavedec(signal_segment, wavelet, level=3)
        cA3, cD3, cD2, cD1 = coeffs  # 分别对应第3级近似系数和第3、2、1级细节系数

        # 生成GBT所需正交矩阵
        n = len(cA3)  # n为近似系数的长度
        A = generate_adjacency_matrix(n)  # 邻接矩阵A
        # 计算每个节点的度数
        degrees = np.sum(A, axis=1)
        # 构造度矩阵 D
        D = np.diag(degrees)
        K = D - A
        # 使用 np.linalg.eig 进行特征值分解
        vals, vecs = np.linalg.eig(K)
        # GBT变换
        G = cA3 @ np.linalg.inv(vecs)
        Gd = np.diag(G)
        # SVD分解
        U, C, VT = np.linalg.svd(Gd)
        C_embed = np.copy(C)

        # 水印嵌入
        C_embed[0] = C[0] + _lambda * W[i]

        # SVD还原
        C_embed_diag = np.diag(C_embed)
        G_embed = U @ C_embed_diag @ VT
        G_embed = np.diag(G_embed)
        # IGBT还原
        D_embed = G_embed @ vecs

        coeffs_new = [D_embed, cD3, cD2, cD1]
        # IDWT还原
        wm_signal_segment = pywt.waverec(coeffs_new, wavelet)
        wm_signal[i * 100:(i + 1) * 100] = wm_signal_segment



    return wm_signal

def ss_watermark_extract(wm_signal, signal,W,_lambda):
    w_extract = np.zeros(W.shape)

    for i in range(len(W)):
        s = wm_signal[i * 100:(i + 1) * 100]
        C_s = DGS_parameter(s)
        C_ori = DGS_parameter(signal[i * 100:(i + 1) * 100])
        w_extract[i] = (C_s[0] - C_ori[0]) / _lambda
    # 像素 四舍五入
    w_extract = np.round(w_extract)
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

# if __name__ == '__main__':

    # _lambda = 2
    # # 三级DWT变换
    # # 生成示例信号
    # t = np.linspace(0, 1, 1000, endpoint=False)
    # signal = 10 * np.sin(2 * np.pi * 5 * t) + np.random.randn(t.size) * 0.5  # 添加噪声
    # wm_signal = np.zeros_like(signal)
    # W = [1, 0, 1, 0, 1, 1, 0, 0, 0, 0]
    # for i in range(len(W)):
    #     wm_signal[i * 100:(i + 1) * 100] = watermark_embed(signal[i * 100:(i + 1) * 100], W[i], _lambda)
    #
    # mse = np.mean((signal - wm_signal) ** 2)
    #
    # w_extract = np.zeros_like(W)
    # for i in range(len(W)):
    #     s = wm_signal[i*100:(i+1)* 100]
    #     C_s = DGS_parameter(s)
    #     C_ori = DGS_parameter(signal[i * 100:(i + 1) * 100])
    #     w_extract[i] = (C_s[0] - C_ori[0]) / _lambda
    #
    # if np.array_equal(w_extract, W):
    #     print("correct")
    # else:
    #     print("wrong")
    #
    # print(mse)
