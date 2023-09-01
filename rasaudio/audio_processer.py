import numpy as np
from pydub import AudioSegment, effects, silence
import io
import scipy.io.wavfile
import array


def normalize(audio_segment, kind='pydub'):
    if kind == 'pydub':
        return effects.normalize(audio_segment)
    
    elif kind == 'jukebox':
        np_segment = np.array(audio_segment.get_array_of_samples(), dtype=np.float16).reshape((-1, segment.channels)).T
        
        if segment.ndim == 1:
            segment = segment[np.newaxis]
        np_segment = np_segment.mean(axis=0)
        
        # normalize audio
        norm_factor = np.abs(np_segment).max()
        if norm_factor > 0:
            np_segment /= norm_factor
        np_segment = np_segment.flatten()
        
        # wav_io = io.BytesIO()
        # scipy.io.wavfile.write(wav_io, 16000, np_segment)
        # wav_io.seek(0)
        # return AudioSegment.from_wav(wav_io)
        array_segment = array.array(audio_segment.array_type, np_segment)
        new_sound = audio_segment._spawn(array_segment)

def remove_silence(audio_segment):
    silence_list = silence.detect_silence()
    
def denoise():
    pass

def super_resolution():
    pass
