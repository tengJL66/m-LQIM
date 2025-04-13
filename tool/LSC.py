import numpy as np
import random
import librosa
import soundfile as sf


def lsc_attack(audio, cropping_ratio, num_crops=3):
    """
    模拟大规模裁剪攻击
    :param audio: 原始音频信号
    :param cropping_ratio: 裁剪比例（0到1之间的值）
    :param num_crops: 裁剪片段的数量
    :return: 受攻击的音频信号+裁剪的片段定位
    """
    # 计算裁剪的总样本数
    total_samples = len(audio)
    total_cropped_samples = int(total_samples * cropping_ratio)

    # 计算每个裁剪片段的样本数
    samples_per_crop = total_cropped_samples // num_crops
    domain = len(audio) // num_crops
    crop_starts = np.zeros(num_crops,dtype=int)
    for i in range(num_crops):
        # 随机选择裁剪的起始位置
        crop_starts[i] = random.sample(range(i * domain, (i + 1) * domain - samples_per_crop), 1)[0]

    # 删除指定范围内的音频片段
    attacked_audio = []
    last_end = 0
    for start in crop_starts:
        attacked_audio.extend(audio[last_end:start])
        last_end = start + samples_per_crop

    attacked_audio.extend(audio[last_end:])

    print("裁剪率:",1-len(attacked_audio)/len(audio))
    return np.array(attacked_audio),crop_starts


if __name__ == '__main__':
    # 加载音频文件
    audio_path = 'D:/Mr.Liu/project/LSC-watermark/blues001.wav'
    y, sr = librosa.load(audio_path, sr=None)

    # 定义裁剪比例
    cropping_ratio = 0.3  # 例如裁剪30%的音频

    # 应用LSC攻击
    attacked_audio, crop_starts= lsc_attack(y, cropping_ratio, num_crops=5)

    # 保存受攻击的音频
    output_path = 'D:/Mr.Liu/project/LSC-watermark/attacked_audio.wav'
    sf.write(output_path, attacked_audio, sr)
    print(len(attacked_audio)/len(y))
