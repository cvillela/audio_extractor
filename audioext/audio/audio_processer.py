import numpy as np
from pydub import AudioSegment, effects, silence
import io
import scipy.io.wavfile
   

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
    np_segment = np_segment / (1 << 8*2 - 1)  # normalization. AudioSegment use int16, so the max value is  `1 << 8*2 - 1`.
    return np_segment

def audiosegment_to_ndarray_32_official(audio_segment):
    # CONVERSION DOES NOT WORK WITH IPD.AUDIO
    channel_sounds = audio_segment.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    return fp_arr


def ndarray32_to_audiosegment(fp_arr,frame_rate):
    wav_io = io.BytesIO()
    scipy.io.wavfile.write(wav_io, frame_rate, fp_arr)
    wav_io.seek(0)
    sound = AudioSegment.from_wav(wav_io)
    return sound
    

def remove_silence(audio_segment, min_silence_len=1000, silence_thresh=-16, seek_step=1):
    return silence.detect_silence(
                        audio_segment=audio_segment,
                        min_silence_len=min_silence_len,
                        silence_thresh=silence_thresh,
                        seek_step=seek_step
                    )
    
def denoise():
    pass

def super_resolution():
    pass
