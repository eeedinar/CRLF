import librosa
import numpy as np
import cv2 as cv
import os
from IPython.display import Audio
import matplotlib.pyplot as plt
import librosa.display


### Misc functions
def cwd_files_search_with(seek_str, search_where = 'end', directory = os.getcwd()):
    """
        files_sorted = cwd_files_search_with('.h5')
    """
    directory = os.getcwd() if directory == None else directory
    files = []
    if search_where == 'end':
        for file in [each for each in os.listdir(directory) if each.endswith(seek_str)]:
            files.append(file)
    
    elif search_where == 'start':
        for file in [each for each in os.listdir(directory) if each.startswith(seek_str)]:
            files.append(file)

    files_sorted = sorted(files)
    return files_sorted

def flatten(S):
    """
    l = [2,[[1,2]],1]
    list(flatten(l))
    """
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


def signal_detection(filename, sr, fmax, n_mels, dB_thr, freq_idx, n_serials, block, show_plot=False, show_results=False, play_audio = 0 ):
    # f[freq_idx], t[r>n_serials] = signal_detection(filename= filename, sr= sr, fmax=fmax, n_mels=n_mels, dB_thr=dB_thr, freq_idx=freq_idx, n_serials=n_serials, block=block, show_plot=False, show_results=False, play_audio = 0 )
    y, sr  = librosa.load(filename, sr = sr)
    S      = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax, power=2.0,)
    t      = np.linspace(start=0, stop=y.shape[0]/sr, num=S.shape[1])
    f      = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=fmax, htk=False)
    S_dB   = librosa.power_to_db(S, ref=np.max)
    S_dB_thr = S_dB > dB_thr
    kernel = np.ones((block,block),np.uint8)
    S_closing = cv.morphologyEx(np.uint8(S_dB_thr), cv.MORPH_CLOSE, kernel)

    if show_plot:
        fig, axs = plt.subplots(ncols=3, nrows=1, sharex=True, sharey=True, figsize=(16,6))
        img     = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=fmax, ax=axs[0])
        fig.colorbar(img, ax=axs[0], format='%+2.0f dB')
        axs[0].set(title='Mel-frequency spectrogram')
        img     = librosa.display.specshow(S_dB_thr, x_axis='time', y_axis='mel', sr=sr, fmax=fmax, ax=axs[1])
        img     = librosa.display.specshow(S_closing, x_axis='time', y_axis='mel', sr=sr, fmax=fmax, ax=axs[2])



    S_block = S_closing[freq_idx,:]  # #np.array([ 0   ,  0  ,   0   ,  0  ,   0  ,   0    , 0  ,   0   ,  0  ,   1  ,   1  ,   1  ,   1   ,  1   ,  0   ,  0   ,  0  ,   0   ,  0  ,   0 ])
    k = np.where( np.insert(np.diff(S_block)!=0, 0, True) )[0]  # insert true first an then find diff to find changes of indices --> array([ 0,  9, 14])
    r = np.zeros_like(S_block)
    tmp = k.tolist()
    tmp.append(len(S_block))
    r[k] = S_block[k]*np.array(np.diff(tmp))

    if show_results:
        print(f'frequency {f[freq_idx]}, and time {t[r>n_serials]}')
    if play_audio:
        Audio(data=y, rate=sr)
    return f[freq_idx], t[r>n_serials]