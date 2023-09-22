import os
import numpy as np
import librosa
from pydub import AudioSegment
from scipy.signal import freqz
import matplotlib.pyplot as plt

from .audio_processer import get_butter_bandpass


def list_wavs_from_dir(path):
    file_paths = []
    for dirpath, _, filenames in os.walk(path):
        for file_name in filenames:
            if file_name.endswith(".wav"):
                file_paths.append(os.path.join(dirpath, file_name))
    return file_paths


def get_len_wavs(file_paths):
    dur = 0
    for f in file_paths:
        audio = AudioSegment.from_file(f)
        dur += len(audio)

    return dur / (1000 * 60 * 60)  # in hours


def plot_spectrogram(
    Y,
    sr,
    hop_length=512,
    y_axis="linear",
    title="",
    save=False,
    filename="",
    highlighted_areas=None,
):
    fig, ax = plt.subplots(figsize=(10, 5))
    img = librosa.display.specshow(
        Y, hop_length=hop_length, x_axis="time", y_axis=y_axis, sr=sr
    )
    fig.colorbar(img, format="%+2.f dB")
    ax.set_title(title, fontsize=20)

    if highlighted_areas is not None:
        for i, area in enumerate(highlighted_areas):
            ax.axvspan(
                area[0],
                area[1],
                facecolor="r",
                alpha=0.3,
                label="_" * i + "perceived bird song",
            )

    if save:
        plt.savefig("./" + filename + title + ".png")


def plot_time(
    signal, sr, x_lim=[None, None], y_lim=[None, None], highlighted_areas=None
):
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))

    x = [i / sr for i in range(len(signal))]
    y = signal

    ax.plot(x, y)
    ax.set_xlabel("Tempo [s]", size="large")
    ax.set_ylabel("Amplitude", size="large")

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    if highlighted_areas is not None:
        for i, area in enumerate(highlighted_areas):
            ax.axvspan(
                area[0],
                area[1],
                facecolor="r",
                alpha=0.3,
                label="_" * i + "perceived bird song",
            )
    plt.legend()
    plt.setp(ax.get_xticklabels(), rotation=0, horizontalalignment="right")

    plt.show()


def plot_fft(signal, sr, title="FFT"):
    fig, ax = plt.subplots(figsize=(10, 5))

    signal -= signal.mean()
    fft = np.fft.fft(signal)

    # take only first half
    fft = fft[: int(len(fft) / 2)]

    # remove 0Hz component - ct
    mag = np.abs(fft)
    freq = np.linspace(0, sr, len(mag))

    ax.set_title(title)
    plt.plot(freq, mag)
    plt.xlabel("Frequência [Hz]")
    plt.ylabel("Amplitude")


def plot_bp_filter(
    sr,
    order=6,
    low=1000,
    high=11000,
):
    a, b = get_butter_bandpass(low, high, sr, order)
    w, h = freqz(b, a, fs=sr, worN=8000)
    plt.subplot(2, 1, 1)
    plt.plot(w, np.abs(h), "b")
    plt.plot(low, 0.5 * np.sqrt(2), "ko")
    plt.axvline(low, color="k")
    plt.plot(high, 0.5 * np.sqrt(2), "ro")
    plt.axvline(high, color="r")
    plt.xlim(0, 0.5 * sr)
    plt.title("Bandpass Filter Frequency Response")
    plt.xlabel("Frequency [Hz]")
    plt.grid()
