import numpy as np
from pydub import AudioSegment, effects, silence
import io
import scipy.io.wavfile
import librosa
from scipy.signal import butter, lfilter, freqz
from pydub.silence import split_on_silence, detect_nonsilent
import noisereduce as nr
import itertools

from .audio_utils import plot_bp_filter


def pcm2float(sig, dtype="float32"):
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    See Also
    --------
    float2pcm, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in "iu":
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max


def normalize_unit(y):
    # ONLY SUPPORTS MONO
    norm_factor = np.abs(y).max()
    if norm_factor > 0:
        y /= norm_factor
    y = np.clip(y, -1.0, 1.0)
    return y


def normalize_loudness(audio_segment):
    return effects.normalize(audio_segment)


def set_channels(audio_segment, n_channels):
    return audio_segment.set_channels(n_channels)


def resample_segment(audio_segment, target_sr):
    return audio_segment.set_frame_rate(target_sr)


def get_dbfs(audio_segment):
    return audio_segment.dBFS


def get_rms(audio_segment):
    return audio_segment.rms


def audiosegment_to_ndarray_32(audio_segment):
    # ONLY SUPPORTS MONO
    np_segment = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    np_segment = np_segment / (
        1 << 8 * 2 - 1
    )  # normalization. AudioSegment use int16, so the max value is  `1 << 8*2 - 1`.
    return np_segment


def audiosegment_to_ndarray_32_official(audio_segment):
    # CONVERSION DOES NOT WORK WITH IPD.AUDIO
    channel_sounds = audio_segment.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    return fp_arr


def ndarray32_to_audiosegment(fp_arr, frame_rate):
    wav_io = io.BytesIO()
    scipy.io.wavfile.write(wav_io, frame_rate, fp_arr)
    wav_io.seek(0)
    sound = AudioSegment.from_wav(wav_io)
    return sound


def remove_silence(
    audio_segment, min_silence_len=1000, silence_thresh=-16, seek_step=1
):
    return silence.detect_silence(
        audio_segment=audio_segment,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        seek_step=seek_step,
    )


def denoise():
    pass


def super_resolution():
    pass


def get_freq_domain(y, frame_size=2048, hop_size=512):
    Y = librosa.stft(y, n_fft=frame_size, hop_length=hop_size)
    Y_db = librosa.amplitude_to_db(
        np.abs(Y), ref=np.max
    )  # convert signal amplitude to DB
    return Y_db


def get_mel(y, sr, frame_size=2048, hop_size=512, n_mels=128):
    Y_mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=frame_size, hop_length=hop_size, n_mels=n_mels
    )
    Y_mel = librosa.amplitude_to_db(np.abs(Y_mel), ref=np.max)
    return


def get_butter_bandpass(lowcut, highcut, sr, order=5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return a, b


def bandpass_filter_signal(y, sr, order=6, low=1000, high=11000, plot=False):
    a, b = get_butter_bandpass(low, high, sr, order=order)
    y_bp = lfilter(b, a, y)

    # Plot the frequency response.
    if plot:
        plot_bp_filter(sr, a, b, low, high)

    return y_bp.astype(np.float32)


def reduce_noise(y, sr, stationary=True):
    return nr.reduce_noise(y=y, sr=sr, stationary=stationary)


def remove_silence(audio_segment):
    audio_chunks = split_on_silence(
        audio_segment,
        min_silence_len=2000,
        silence_thresh=-45,
        keep_silence=1000,
    )

    return sum(audio_chunks)


def get_silence_ranges(
    audio_segment,
    min_silence_len=1000,
    silence_thresh=-16,
    keep_silence=100,
    seek_step=1,
):
    if isinstance(keep_silence, bool):
        keep_silence = len(audio_segment) if keep_silence else 0

    output_ranges = [
        [start - keep_silence, end + keep_silence]
        for (start, end) in detect_nonsilent(
            audio_segment, min_silence_len, silence_thresh, seek_step
        )
    ]

    return output_ranges


def split_on_silence(audio_segment, output_ranges):
    # from the itertools documentation
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    for range_i, range_ii in pairwise(output_ranges):
        last_end = range_i[1]
        next_start = range_ii[0]
        if next_start < last_end:
            range_i[1] = (last_end + next_start) // 2
            range_ii[0] = range_i[1]

    return [
        audio_segment[max(start, 0) : min(end, len(audio_segment))]
        for start, end in output_ranges
    ]
