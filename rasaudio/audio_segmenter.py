
import os
from pydub import AudioSegment, silence, effects
import uuid
from unidecode import unidecode
import json
import numpy as np


def get_audio_metadata(audio):
    meta =  {
        "sample_rate": audio.frame_rate,
        "channels": audio.channels, 
        "bytes_per_sample":audio.sample_width, # 1 means 8 bit, 2 means 16 bit   
    }

    return meta


def save_sample_meta(audio_meta, segment, out_dir):
    
    # declare metadata and sample dirs
    sample_dir = out_dir + '/samples/'
    meta_dir = out_dir + '/metadata/'
    
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(meta_dir):
        os.makedirs(meta_dir)
    
    sample_name = str(uuid.uuid4())
    seg_path = os.path.join(sample_dir, sample_name+'.wav')
    meta_path = os.path.join(meta_dir, sample_name+'.json')
    
    # here we can insert sample specific information on the metadata
    segment_meta_dict = audio_meta.copy()
    
    # Save the segment
    segment.export(seg_path, format='wav')
    
    # Save the metadata
    with open(meta_path, 'w', encoding='utf8') as fp:
        json.dump(segment_meta_dict, fp)
        
        
def segment_audio(
        file_path, out_dir, segment_length_s=10, target_sr=32000, n_channels=1,
        cutoff='pad', overlap=0.0, normalize=False, denoise=False, desilence=False
    ):

    # Get file name without extension for caption
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    file_name = unidecode(file_name.lower())
    
    # load audio
    audio = AudioSegment.from_file(file_path)
    
    # resample
    if target_sr is not None:
        audio = audio.set_frame_rate(target_sr) # resample to target_sr
    
    # force mono/stereo          
    if n_channels is not None:
        audio = audio.set_channels(n_channels) # convert to mono
    
    # remove noise
    
    # remove silence
    if desilence:
        audio = remove_silence(audio)
    
    # Audio info dict
    audio_metadata = get_audio_metadata(audio, file_name) 
    audio_metadata["title"] = file_name.split('.')[0]
    
    # segment into segment_length_s samples
    segment_length_ms = segment_length_s * 1000
    step = int((1-overlap)*segment_length_ms)
    for i in range(0, len(audio), step):
        # create segment
        start_time = i 
        end_time = i + segment_length_ms
        segment = audio[start_time:end_time]

        # pad or crop end-of-file samples
        if end_time > len(audio):
            if cutoff=='pad': # pad with silence (0 amplitude)
                if len(segment) < 0.5*segment_length_ms: # pad at most 50% of the signal
                    break
                pad_len = segment_length_ms - len(segment)
                silence = AudioSegment.silent(  
                    duration=pad_len,
                    frame_rate=segment.frame_rate
                )
                segment = segment + silence
            
            elif cutoff=='leave': # save sample smaller than segment_length_s
                segment = audio[start_time:len(audio)] 
            
            elif cutoff=='crop': # discard smaller sample
                break
            
        save_sample_meta(out_dir)