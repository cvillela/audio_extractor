import numpy as np
from pydub import AudioSegment, effects, silence
import io
import scipy.io.wavfile
import array


def remove_silence(audio_segment):
    silence_list = silence.detect_silence()
    

def normalization_pipeline(audio_segment, pipeline=None):
    
    # assert kind is not none and raise custom message if not 'pydub' or 'jukebox'
    
    assert pipeline is not None, "You must declare the normalization pipeline."
    
    if pipeline == 'pydub':
        return effects.normalize(audio_segment)
    
    elif pipeline == 'jukebox':
        # ONLY SUPPORTS MONO
        array_segment = normalize_unit(audio_segment.get_array_of_samples())        
        new_segment = audio_segment._spawn(array_segment)
        return new_segment

def normalize_unit(y):
    # ONLY SUPPORTS MONO
    norm_factor = np.abs(y).max()
    if norm_factor > 0:
        y /= norm_factor
    return y


def audiosegment_to_ndarray_32(audio_segment):
    # ONLY SUPPORTS MONO
    # np_segment = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    # np_segment = np_segment / (1 << 8*2 - 1)  # normalization. AudioSegment use int16, so the max value is  `1 << 8*2 - 1`.
    # return np_segment
    channel_sounds = audio_segment.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    return fp_arr


def ndarray32_to_audiosegment(fp_arr,frame_rate):
    wav_io = io.BytesIO()
    scipy.io.wavfile.write(wav_io, 16000, fp_arr)
    wav_io.seek(0)
    sound = AudioSegment.from_wav(wav_io)
    return sound
    

def remove_silence(audio_segment):
    silence_list = silence.detect_silence()
    
def denoise():
    pass

def super_resolution():
    pass
