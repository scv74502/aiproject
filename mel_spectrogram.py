from scipy.io import wavfile
from scipy.signal import stft
import matplotlib.pyplot as plt
import librosa
import numpy as np
import librosa.display
import os

def Mel_S(wav_file, file_name):
    # mel-spectrogram
    y, sr = librosa.load(wav_file, sr=16000)

    # wav_length = len(y)/sr
    input_nfft = int(round(sr * frame_length))
    input_stride = int(round(sr * frame_stride))

    S = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nfft, hop_length=input_stride)

    print("Wav length: {}, Mel_S shape:{}".format(len(y) / sr, np.shape(S)))



    plt.figure(figsize=(10, 4))
    # librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', sr=sr, hop_length=input_stride,
                             # x_axis='time')
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, hop_length=input_stride)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.savefig('./bus_mel_spectrogram/{}.png'.format(file_name))
    print(S.shape)

    return S

path = "./busWAV/"
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith(".WAV")]
print("file_list_py: {}".format(file_list_py))
frame_length = 0.025
frame_stride = 0.010

for i, file in enumerate(file_list_py):
    wav_file = path + file
    mel_spec = Mel_S(wav_file, i)

end = 1
