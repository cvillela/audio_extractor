import os
import numpy as np
import librosa
from pydub import AudioSegment
from scipy.signal import freqz
import matplotlib.pyplot as plt


def list_wavs_from_dir(path, walk=True):
    file_paths = []

    if walk:
        for dirpath, _, filenames in os.walk(path):
            for file_name in filenames:
                if (
                    file_name.lower().endswith(".wav")
                    or file_name.lower().endswith(".mp3")
                    or file_name.lower().endswith(".flac")
                    or file_name.lower().endswith(".m4a")
                ):
                    file_paths.append(os.path.join(dirpath, file_name))
    else:
        for item in os.listdir(path):
            if os.path.isfile(os.path.join(path, item)) and (
                item.lower().endswith(".wav")
                or item.lower().endswith(".mp3")
                or item.lower().endswith(".flac")
                or item.lower().endswith(".m4a")
            ):
                file_paths.append(os.path.join(path, item))

    return file_paths


def get_len_wavs(file_paths):
    dur = 0
    for f in file_paths:
        audio = AudioSegment.from_file(f)
        dur += len(audio)

    return dur / (1000 * 60 * 60)  # in hours


def get_audio_segment_size_gb(audio_segment):
    duration_seconds = len(audio_segment) / 1000  # Convert milliseconds to seconds
    frame_rate = audio_segment.frame_rate
    sample_width = audio_segment.sample_width
    num_channels = audio_segment.channels

    size_bytes = duration_seconds * frame_rate * sample_width * num_channels
    size_gb = size_bytes / (1024 ** 3)  # Convert bytes to gigabytes

    return size_gb


def linear_crossfade(a1, a2, crossfade_duration, sample_rate):
    """
    Crossfades two audio signals.

    Parameters:
    audio1 (numpy array): First audio waveform.
    audio2 (numpy array): Second audio waveform.
    crossfade_duration (float): Crossfade duration in seconds.
    sample_rate (int): Sample rate of the audio signals.

    Returns:
    numpy array: Crossfaded audio signal.
    """

    # Number of samples over which to crossfade
    crossfade_samples = int(crossfade_duration * sample_rate)

    assert (
        a1.ndim >= 2 and a2.ndim >= 2
    ), "Array must be at least 2D, of shape [N_CHANNELS, N_SAMPLES]"
    assert a1.shape[0] == a2.shape[0], "Arrays must have the same number of channels."

    n_channels = a1.shape[0]

    # Ensure audio1 is long enough for crossfade
    if len(a1[0]) < crossfade_samples:
        return a2
    # Ensure audio2 is long enough for crossfade
    if len(a2[0]) < crossfade_samples:
        return a1

    # Create multichannel fade windows
    fade_out_curve = np.linspace(1, 0, crossfade_samples)
    fade_out_curve = np.array([fade_out_curve] * n_channels).astype(a1[0].dtype)

    fade_in_curve = np.linspace(0, 1, crossfade_samples)
    fade_in_curve = np.array([fade_in_curve] * n_channels).astype(a2[0].dtype)

    # end of input audio will fade out on last crossfade samples
    a1[:, -crossfade_samples:] = np.multiply(a1[:, -crossfade_samples:], fade_out_curve)
    # fade in on first crossfade samples of a2
    fade_in = np.multiply(a2[:, :crossfade_samples], fade_in_curve)
    # sum faded in audio from a2 to a1
    a1[:, -crossfade_samples:] = np.add(
        a1[:, -crossfade_samples:],
        fade_in,
    )

    # remove first crossfade samples of a2
    a2 = a2[:, crossfade_samples:]

    # return concatenated a1 and a2
    return np.concatenate((a1, a2), axis=1)


def remove_intervals(data, intervals):
    """
    Remove intervals from a 2D numpy array along axis=1.

    :param data: 2D numpy array of shape [N-Channels, N_Samples]
    :param intervals: List of [start, end] pairs defining intervals to remove
    :return: 2D numpy array with specified intervals removed
    """
    # Number of samples
    num_samples = data.shape[1]

    # Create an initial mask filled with True
    mask = np.ones(num_samples, dtype=bool)

    # Update the mask to False for indices in the specified intervals
    for start, end in intervals:
        mask[start:end] = False

    # Apply the mask to the data
    return data[:, mask]


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
    a,
    b,
    low=1000,
    high=11000,
):
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
