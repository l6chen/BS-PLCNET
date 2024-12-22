from torch.utils.data import Dataset
import torchaudio as ta
import torch
from utils import *


def calculate_energy(audio_signal):
    """
    计算音频信号的能量
    :param audio_signal: 一维的音频信号数据（numpy数组格式）
    :return: 音频信号的能量值
    """
    return torch.sum(torch.pow(audio_signal, 2))


def vad_based_on_energy(wav, window_size=0.02, step_size=0.01, energy_threshold=0.1, sample_rate=16000):
    """
    基于音频时域能量的VAD算法
    :param audio_path: 音频文件的路径
    :param window_size: 分析窗口大小（单位：秒），默认0.02秒
    :param step_size: 窗口移动步长（单位：秒），默认0.01秒
    :param energy_threshold: 能量阈值，用于判断是否有语音，可根据实际情况调整，默认0.1
    :return: 语音活动的检测结果（布尔值列表，True表示对应时段有语音，False表示无语音）
    """
    # 加载音频文件，获取音频信号和采样率

    audio_signal = wav[0]
    window_samples = int(window_size * sample_rate)
    step_samples = int(step_size * sample_rate)
    num_frames = 1 + (len(audio_signal) - window_samples) // step_samples
    results = []
    for i in range(num_frames):
        start = i * step_samples
        end = start + window_samples
        frame_signal = audio_signal[start:end]
        energy = calculate_energy(frame_signal)
        is_voice = 1 if energy > energy_threshold else 0
        results.append(is_voice)
    vad_prob = sum(results) / len(results)
    return results, vad_prob


class PLCDataset(Dataset):
    def __init__(self, path_dir, seg_len, mode='train'):
        self.seg_len = seg_len
        self.path_dir = path_dir
        self.paths_wav = get_wav_paths_plain(self.path_dir)
        self.mode = mode

    def __len__(self):
        return len(self.paths_wav)

    def __getitem__(self, idx):
        wav, sr = ta.load(self.paths_wav[idx])
        wav_length = wav.shape[1]
        while wav_length < 3 * 16000:
            wav = torch.cat((wav, wav), dim=1)
            wav_length = wav.shape[1]
        start = int(torch.randint(0, wav_length - self.seg_len * 16000, (1,)))
        wav = wav[:, start:start + self.seg_len * 16000]
        wav_label = wav.detach().clone()
        prob_loss = 1.1
        loss_len = int(torch.randint(0, 100, (1,)) * sr / 1000)# ms
        if torch.rand(1) < prob_loss:
            loss_point = int(torch.randint(0, 20, (1,)))
            loss_segs = [int(self.seg_len * 16000 * i / loss_point) for i in range(loss_point)]
            for i, _ in enumerate(loss_segs[:-1]):
                wav_seg = wav[:, loss_segs[i] : loss_segs[i + 1]]
                if vad_based_on_energy(wav_seg)[1] < 0.5:
                    continue
                loss_start = torch.randint(0, loss_segs[i + 1] - loss_len, (1,))
                wav[:, loss_start: loss_len] = 0.0
        return wav, wav_label