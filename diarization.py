from pyannote.audio import Pipeline
import librosa
import numpy as np
import parselmouth
import torch
#from utils import audio

def f0_to_coarse(f0):
    f0_bin = 256
    f0_max = 1100.0
    f0_min = 50.0
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    
    is_torch = isinstance(f0, torch.Tensor)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(np.int_)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse

def librosa_pad_lr(x, fsize, fshift, pad_sides=1):
    '''compute right padding (final frame) or both sides padding (first and final frames)
    '''
    assert pad_sides in (1, 2)
    # return int(fsize // 2)
    pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
    if pad_sides == 1:
        return 0, pad
    else:
        return pad // 2, pad // 2 + pad % 2
    
# Conversions
def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def normalize(S, hparams):
    return (S - hparams['min_level_db']) / -hparams['min_level_db']

def process_utterance(wav_path,
                      fft_size=1024,
                      hop_size=256,
                      win_length=1024,
                      window="hann",
                      num_mels=80,
                      fmin=80,
                      fmax=7600,
                      eps=1e-6,
                      sample_rate=22050,
                      loud_norm=False,
                      min_level_db=-100,
                      return_linear=False,
                      trim_long_sil=False, vocoder='pwg'):
    if isinstance(wav_path, str):
        wav, _ = librosa.core.load(wav_path, sr=sample_rate)
    else:
        wav = wav_path

    # if loud_norm:
    #     meter = pyln.Meter(sample_rate)  # create BS.1770 meter
    #     loudness = meter.integrated_loudness(wav)
    #     wav = pyln.normalize.loudness(wav, loudness, -22.0)
    #     if np.abs(wav).max() > 1:
    #         wav = wav / np.abs(wav).max()

    # get amplitude spectrogram
    x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="constant")
    spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if fmin == -1 else fmin
    fmax = sample_rate / 2 if fmax == -1 else fmax
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel = mel_basis @ spc

    mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)

    l_pad, r_pad = librosa_pad_lr(wav, fft_size, hop_size, 1)
    wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)
    wav = wav[:mel.shape[1] * hop_size]

    #如果return_linear为False，则返回语音数据wav和梅尔频谱mel。
    #否则，将幅度谱spc转换为分贝表示，并进行分贝归一化，然后返回语音数据wav、梅尔频谱mel和幅度谱spc。
    if not return_linear:
        return wav, mel.T
    else:
        spc = amp_to_db(spc)
        spc = normalize(spc, {'min_level_db': min_level_db})
        return wav, mel.T, spc

def get_pitch(wav_data, mel, sample_rate):
    """

    :param wav_data: [T]
    :param mel: [T, 80]
    :param hparams:
    :return:
    """
    hop_size=256
    #sample_rate=22050
    time_step = hop_size / sample_rate * 1000
    f0_min = 80
    f0_max = 750

    if hop_size == 128:
        pad_size = 4
    elif hop_size == 256:
        pad_size = 2
    else:
        assert False

    f0 = parselmouth.Sound(wav_data, sample_rate).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
    #lpad = pad_size * 2
    lpad = pad_size
    rpad = len(mel) - len(f0) - lpad
    f0 = np.pad(f0, [[lpad, rpad]], mode='constant')
    # mel and f0 are extracted by 2 different libraries. we should force them to have the same length.
    # Attention: we find that new version of some libraries could cause ``rpad'' to be a negetive value...
    # Just to be sure, we recommend users to set up the same environments as them in requirements_auto.txt (by Anaconda)
    delta_l = len(mel) - len(f0)
    assert np.abs(delta_l) <= 8
    if delta_l > 0:
        f0 = np.concatenate([f0, [f0[-1]] * delta_l], 0)
    f0 = f0[:len(mel)]
    pitch_coarse = f0_to_coarse(f0)
    return f0, pitch_coarse

def get_data(wav, diarization):

    audio_data, sample_rate = librosa.load(wav)
    clip = []
    duration = []
    spk = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        clip.append([int(sample_rate * turn.start), int(sample_rate * turn.end)])
        duration.append(turn.duration)
        spk.append(speaker)
    dict = {}
    clip_dict = {}
    f0_dict = {}
    pitch_dict = {}
    for i, name in enumerate(spk):
        if name in dict:
            dict[name][0].append(clip[i])
            dict[name][1] += duration[i]
        else:
            dict[name] = [[clip[i]], duration[i]]
    for spk in dict:
        clip_dict[spk] = np.concatenate([audio_data[i[0]:i[1]] for i in dict[spk][0]])
        clip_wav, mel = process_utterance(wav_path=clip_dict[spk], sample_rate=sample_rate)
        f0, pitch_dict[spk] = get_pitch(clip_wav, mel, sample_rate)
        f0_dict[spk] = np.mean(f0[f0>50])

    #得到clip，计算f0，取f0最大的spk_max
    max_f0 = max(f0_dict.values())  # 获取最大值
    max_keys = [key for key, value in f0_dict.items() if value == max_f0]
    
    pitch = pitch_dict[max_keys[0]]
    # # dump the diarization output to disk using RTTM format
    # with open("audio.rttm", "w") as rttm:
    #     diarization.write_rttm(rttm)
    
    return dict[max_keys[0]][1], pitch, max_f0
    #return total, pitch_distribution, highest_f0

def main():
    # 1. visit hf.co/pyannote/speaker-diarization and accept user conditions
    # 2. visit hf.co/pyannote/segmentation and accept user conditions
    # 3. visit hf.co/settings/tokens to create an access token
    # 4. instantiate pretrained speaker diarization pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                        use_auth_token="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")


    # apply the pipeline to an audio file
    #wav = "/home/lab-su.di/huggingface/ref_audio/baby/baby6.wav"
    wav = "/home/lab-su.di/diffspeech/ref_audio/I say neither yea nor nay.wav"
    diarization = pipeline(wav, num_speakers=1)

    total, pitch, highest_f0 = get_data(wav, diarization)

    total_duration = 0
    pitch_distribution = 0
    #longest_phoneme = 'u'

    child_speaker = 0
    if highest_f0 > 250:
        child_speaker = 1

    if child_speaker:
        total_duration = total
        pitch_distribution = pitch
    
    print('wav:', wav)
    print('total_duration:', total_duration)
    #print('pitch_distribution:', pitch_distribution)
    print('highest_f0:', highest_f0)

if __name__ == '__main__':
    main()