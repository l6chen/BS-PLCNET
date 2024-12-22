import os
import torch
def get_wav_paths_plain(paths: str):
    with open(paths, 'r') as fin:
        flist = fin.readlines()
    wav_paths = [i.replace('\n', '') for i in flist]
    wav_paths.sort(key=lambda x: os.path.split(x)[-1])
    return wav_paths

def lsd_distance(magnitude_spectrogram1, magnitude_spectrogram2):

    # 对幅度谱取对数
    log_magnitude_spectrogram1 = torch.log10(magnitude_spectrogram1 + 1e-10)
    log_magnitude_spectrogram2 = torch.log10(magnitude_spectrogram2 + 1e-10)

    # 计算平方差
    squared_diff = torch.pow(log_magnitude_spectrogram1 - log_magnitude_spectrogram2, 2)

    # 计算每帧的均方根（RMS）
    rms_per_frame = torch.sqrt(torch.mean(squared_diff, dim=0))

    # 计算LSD距离（取平均）
    lsd = torch.mean(rms_per_frame)
    return lsd