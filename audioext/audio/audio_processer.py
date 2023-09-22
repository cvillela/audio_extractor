import numpy as np
from pydub import AudioSegment, effects, silence
import io
import scipy.io.wavfile
import librosa
from scipy.signal import butter, lfilter, freqz
from pydub.silence import split_on_silence
import noisereduce as nr


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


def bandpass_filter_signal(y, sr, order=6, low=1000, high=11000, plot=True):
    b, a = get_butter_bandpass(low, high, sr, order=order)
    y_bp = lfilter(b, a, y)

    # Plot the frequency response.
    # if plot:
    #     # Get the filter coefficients so we can check its frequency response.
    #     plot_bp_filter(sr, order, low, high)

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
