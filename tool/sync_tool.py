

def detect_sync(audio_segment, sync_template):
    # 计算音频段与同步模板的互相关
    correlation = np.correlate(audio_segment, sync_template, mode='same')
    peaks = find_peaks(correlation, height=threshold)
    return peaks


# 对裁剪后的A2遍历检测所有可能同步头
detected_syncs = []
for sync_id in [1-10]:
    template = load_sync_template(sync_id)  # 加载预存的同步头模板
    peaks = detect_sync(A2, template)
    detected_syncs.extend([(pos, sync_id) for pos in peaks])
# 按位置排序并去重，得到有效分块位置