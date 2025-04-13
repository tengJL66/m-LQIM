import pywt
import numpy as np
import math
from lattice_embed.QIM import standard_qim, cosets, qim_decode
from lattice_embed.Lattice.Lattice_basis import Lattice_Basis
from lattice_embed import Create_Data
from scipy.fftpack import dct, idct


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


def qim_watermark_embed(signal, W, b, seg_len, wm_type):
    wm_signal = np.copy(signal)
    N = b.shape[0]
    alpha = 2
    coset_representatives = cosets(alpha, N)

    j = 0  # 音频信号索引
    i = 0  # 水印索引

    while (i < len(W)):
        signal_segment = signal[j:j + seg_len]
        if wm_type=='DWTSVD':
            # dwt+svd
            wavelet = 'db1'  # Daubechies小波
            # 进行三级离散小波变换
            coeffs = pywt.wavedec(signal_segment, wavelet, level=3)
            cA3, cD3, cD2, cD1 = coeffs  # 分别对应第3级近似系数和第3、2、1级细节系数
            U,S,VT = SVD(np.diag(cA3))
            S_embed = np.copy(S)
            S_embed[1:1 + N] = standard_qim(S[1:1 + N], b, W[i], alpha, coset_representatives)
            # 逆变换
            cA3_embed = U @ S_embed @ VT
            coeffs_new = [cA3_embed, cD3, cD2, cD1]
            # IDWT还原
            wm_signal[j:j + seg_len] = pywt.waverec(coeffs_new, wavelet)
        if wm_type=='DWTDCT':
            # dwt+dct
            wavelet = 'db1'  # Daubechies小波
            # 进行三级离散小波变换
            coeffs = pywt.wavedec(signal_segment, wavelet, level=3)
            cA3, cD3, cD2, cD1 = coeffs  # 分别对应第3级近似系数和第3、2、1级细节系数
            D = dct(cA3, type=2, norm="ortho")
            U, S, VT = SVD(np.diag(D))
            S_embed = np.copy(S)
            S_embed[1:1 + N] = standard_qim(S[1:1 + N], b, W[i], alpha, coset_representatives)
            # 逆变换
            D_embed = U @ S_embed @ VT
            cA3_embed = idct(D_embed, 2, norm="ortho")
            coeffs_new = [cA3_embed, cD3, cD2, cD1]
            # IDWT还原
            wm_signal[j:j + seg_len] = pywt.waverec(coeffs_new, wavelet)

        if wm_type=='DCTSVD':
            # dct-svd
            A = dct(signal_segment, type=2, norm="ortho")
            U, S, VT = SVD(np.diag(A))
            S_embed = np.copy(S)
            S_embed[20:20 + N] = standard_qim(S[20:20 + N], b, W[i], alpha, coset_representatives)
            # 逆变换
            A_embed = U @ S_embed @ VT
            wm_signal[j:j + seg_len] = idct(A_embed, 2, norm="ortho")

        if wm_type == 'FFT':
            # fft
            # wavelet = 'db1'  # Daubechies小波
            # # 进行三级离散小波变换
            # coeffs = pywt.wavedec(signal_segment, wavelet, level=3)
            # cA3,cD3, cD2, cD1 = coeffs  # 分别对应第3级近似系数和第3、2、1级细节系数
            fft_result = np.fft.fft(signal_segment)
            fft_magnitude = np.abs(fft_result)  # 幅度
            fft_phase = np.angle(fft_result)  # 相位

            S_embed = np.copy(fft_magnitude)
            S_embed[5:5 + N] = standard_qim(fft_magnitude[5:5 + N], b, W[i], alpha, coset_representatives)
            S_embed[-(5+N):-5] = S_embed[5:5 + N] #保证对称性
            # 逆变换
            fft_result_watermarked = S_embed * np.exp(1j * fft_phase)
            wm_signal[j:j + seg_len] = np.fft.ifft(fft_result_watermarked).real
            # coeffs_new = [cA3_embed, cD3,cD2, cD1]
            # IDWT还原
            # wm_signal[j:j + seg_len] = pywt.waverec(coeffs_new, wavelet)

        if wm_type == 'DWTFFT':
            # fft
            wavelet = 'db1'  # Daubechies小波
            # 进行三级离散小波变换
            coeffs = pywt.wavedec(signal_segment, wavelet, level=3)
            cA3,cD3, cD2, cD1 = coeffs  # 分别对应第3级近似系数和第3、2、1级细节系数
            fft_result = np.fft.fft(cA3)
            fft_magnitude = np.abs(fft_result)  # 幅度
            fft_phase = np.angle(fft_result)  # 相位

            S_embed = np.copy(fft_magnitude)
            S_embed[5:5 + N] = standard_qim(S_embed[5:5 + N], b, W[i], alpha, coset_representatives)
            S_embed[-(5 + N):-5] = S_embed[5:5 + N]  # 保证对称性
            # 逆变换
            fft_result_watermarked = S_embed * np.exp(1j * fft_phase)
            cA3_embed = np.fft.ifft(fft_result_watermarked).real
            coeffs_new = [cA3_embed, cD3,cD2, cD1]
            # IDWT还原
            wm_signal[j:j + seg_len] = pywt.waverec(coeffs_new, wavelet)

        i += 1
        j += seg_len


        # # dct
        # A = dct(signal_segment, type=2, norm="ortho")
        # A_embed = np.copy(A)
        # A_embed[20:20 + N] = standard_qim(A[20:20 + N], b, W[i], alpha, coset_representatives)
        # # 逆变换
        # wm_signal[j:j + seg_len] = idct(A_embed, 2, norm="ortho")

        # # dwt
        # wavelet = 'db1'  # Daubechies小波
        # # 进行三级离散小波变换
        # coeffs = pywt.wavedec(signal_segment, wavelet, level=3)
        # cA3, cD3, cD2, cD1 = coeffs  # 分别对应第3级近似系数和第3、2、1级细节系数
        # cA3_embed = np.copy(cA3)
        # cA3_embed[1:1 + N] = standard_qim(cA3[1:1 + N], b, W[i], alpha, coset_representatives)
        # coeffs_new = [cA3_embed, cD3, cD2, cD1]
        # # IDWT还原
        # wm_signal[j:j + seg_len] = pywt.waverec(coeffs_new, wavelet)





        # # fft
        # # wavelet = 'db1'  # Daubechies小波
        # # # 进行三级离散小波变换
        # # coeffs = pywt.wavedec(signal_segment, wavelet, level=3)
        # # cA3,cD3, cD2, cD1 = coeffs  # 分别对应第3级近似系数和第3、2、1级细节系数
        # fft_result = np.fft.fft(signal_segment)
        # fft_magnitude = np.abs(fft_result)  # 幅度
        # fft_phase = np.angle(fft_result)  # 相位
        #
        # S_embed = np.copy(fft_magnitude)
        # S_embed[5:5 + N] = standard_qim(fft_magnitude[5:5 + N], b, W[i], alpha, coset_representatives)
        # S_embed[-(5+N):-5] = S_embed[5:5 + N] #保证对称性
        # # 逆变换
        # fft_result_watermarked = S_embed * np.exp(1j * fft_phase)
        # wm_signal[j:j + seg_len] = np.fft.ifft(fft_result_watermarked).real
        # # coeffs_new = [cA3_embed, cD3,cD2, cD1]
        # # IDWT还原
        # # wm_signal[j:j + seg_len] = pywt.waverec(coeffs_new, wavelet)

        # corr = np.corrcoef(A, A_embed)[0, 1]
        # print(corr)

        # A = dct(wm_signal[j:j + seg_len], 2, norm="ortho")
        # U, S, VT = SVD(np.diag(A))
        # print(W[i], qim_decode(S[20:20 + N], b, alpha, coset_representatives))


    return wm_signal


def qim_watermark_extract(wm_signal, W, b, seg_len,wm_type):
    # QIM参数
    N = b.shape[0]
    alpha = 2
    coset_representatives = cosets(alpha, N)

    # create secret messages 生成
    w_extract = np.zeros_like(W)
    i = 0
    j = 0
    while (i < len(W)):
        s = wm_signal[j:j + seg_len]
        # #dct-svd提取
        # A = dct(s, 2, norm="ortho")
        # U, S, VT = SVD(np.diag(A))
        # w_extract[i] = qim_decode(S[20:20 + N], b, alpha, coset_representatives)
        if wm_type == "DWTSVD":
            # dwt+svd
            wavelet = 'db1'  # Daubechies小波
            # 进行三级离散小波变换
            coeffs = pywt.wavedec(s, wavelet, level=3)
            cA3, cD3, cD2, cD1 = coeffs  # 分别对应第3级近似系数和第3、2、1级细节系数
            U,S,VT = SVD(np.diag(cA3))
            w_extract[i] = qim_decode(S[1:1 + N], b, alpha, coset_representatives)

        if wm_type == "DWTDCT":
            # dwt+dct
            wavelet = 'db1'  # Daubechies小波
            # 进行三级离散小波变换
            coeffs = pywt.wavedec(s, wavelet, level=3)
            cA3, cD3, cD2, cD1 = coeffs  # 分别对应第3级近似系数和第3、2、1级细节系数
            D = dct(cA3, type=2, norm="ortho")
            U, S, VT = SVD(np.diag(D))
            w_extract[i] = qim_decode(S[1:1 + N], b, alpha, coset_representatives)
        if wm_type == "DCTSVD":
            # dctsvd
            A = dct(s, 2, norm="ortho")
            U, S, VT = SVD(np.diag(A))
            w_extract[i] = qim_decode(S[20:20 + N], b, alpha, coset_representatives)

        if wm_type == "FFT":
            # fft
            # wavelet = 'db1'  # Daubechies小波
            # # 进行三级离散小波变换
            # coeffs = pywt.wavedec(s, wavelet, level=3)
            # cA3, cD3,cD2, cD1 = coeffs  # 分别对应第3级近似系数和第3、2、1级细节系数
            fft_result = np.fft.fft(s)
            fft_magnitude = np.abs(fft_result)  # 幅度
            # D= dct(fft_magnitude,type=2,norm="ortho")
            # U, S, VT = SVD(np.diag(D))
            w_extract[i] = qim_decode(fft_magnitude[5:5 + N], b, alpha, coset_representatives)

        if wm_type == "DWTFFT":
            # fft
            wavelet = 'db1'  # Daubechies小波
            # 进行三级离散小波变换
            coeffs = pywt.wavedec(s, wavelet, level=3)
            cA3, cD3,cD2, cD1 = coeffs  # 分别对应第3级近似系数和第3、2、1级细节系数
            fft_result = np.fft.fft(cA3)
            fft_magnitude = np.abs(fft_result)  # 幅度
            w_extract[i] = qim_decode(fft_magnitude[5:5 + N], b, alpha, coset_representatives)

        i += 1
        j += seg_len


        # # dwt提取
        # wavelet = 'db1'  # Daubechies小波
        # # 进行三级离散小波变换
        # coeffs = pywt.wavedec(s, wavelet, level=3)
        # cA3, cD3, cD2, cD1 = coeffs  # 分别对应第3级近似系数和第3、2、1级细节系数
        # w_extract[i] = qim_decode(cA3[1:1 + N], b, alpha, coset_representatives)






        # # fft
        # # wavelet = 'db1'  # Daubechies小波
        # # # 进行三级离散小波变换
        # # coeffs = pywt.wavedec(s, wavelet, level=3)
        # # cA3, cD3,cD2, cD1 = coeffs  # 分别对应第3级近似系数和第3、2、1级细节系数
        # fft_result = np.fft.fft(s)
        # fft_magnitude = np.abs(fft_result)  # 幅度
        # # D= dct(fft_magnitude,type=2,norm="ortho")
        # # U, S, VT = SVD(np.diag(D))
        # w_extract[i] = qim_decode(fft_magnitude[5:5 + N], b, alpha, coset_representatives)



    if np.array_equal(w_extract, W):
        # print("嵌入后直接提取，结果correct")
        a=1
    else:
        # 计算错误比特数
        error_bits = np.sum(w_extract != W)
        # 计算总比特数
        total_bits = w_extract.size
        # 计算误码率
        ber = error_bits / total_bits
        print("BER:", ber)
        # print("嵌入后直接提取，水印无法完整提取")
        com = np.where(w_extract == W, 1, -1)
    return w_extract
