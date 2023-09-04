
import os
from pydub import AudioSegment, silence, effects
import uuid
from unidecode import unidecode
import json
import numpy as np


def get_audio_metadata(audio):
    """
    Generate the audio metadata for the given audio.

    Parameters:
        audio (AudioSegment): The audio segment for which to retrieve the metadata.

    Returns:
        dict: A dictionary containing the audio metadata with the following keys:
            - sample_rate (int): The sample rate of the audio.
            - channels (int): The number of audio channels.
            - bytes_per_sample (int): The number of bytes per audio sample. A value of 1 indicates 8 bit, and a value of 2 indicates 16 bit.
    """
    meta =  {
        "sample_rate": audio.frame_rate,
        "channels": audio.channels, 
        "bytes_per_sample":audio.sample_width, # 1 means 8 bit, 2 means 16 bit   
    }

    return meta


def save_sample_meta(audio_meta, segment, out_dir):
    """
    Save the audio segment and its metadata to the specified output directory.

    Args:
        audio_meta (dict): Metadata for the audio segment.
        segment (AudioSegment): Audio segment to be saved.
        out_dir (str): Output directory where the segment and metadata will be saved.
    """
    # declare metadata and sample dirs
    sample_dir = os.path.join(out_dir, 'samples')
    meta_dir = os.path.join(out_dir, 'metadata')
    
    # create sample and metadata directories if they don't exist
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    
    sample_name = str(uuid.uuid4())
    seg_path = os.path.join(sample_dir, f'{sample_name}.wav')
    meta_path = os.path.join(meta_dir, f'{sample_name}.json')
    
    # create a copy of audio_meta dictionary to add segment specific information
    segment_meta_dict = audio_meta.copy()
    
    # Save the segment as a WAV file
    segment.export(seg_path, format='wav')
    
    # Save the metadata as a JSON file
    with open(meta_path, 'w', encoding='utf8') as fp:
        json.dump(segment_meta_dict, fp)
        
        
def segment_audio(
        file_path, out_dir, segment_length_s=10, target_sr=32000, n_channels=1,
        cutoff='pad', overlap=0.0, normalize=False, denoise=False, desilence=False
    ):
    """
    Segment an audio file into smaller segments of a specified length.

    Args:
        file_path (str): The path to the audio file.
        out_dir (str): The directory where the segmented audio files will be saved.
        segment_length_s (float, optional): The length of each segment in seconds. Defaults to 10.
        target_sr (int, optional): The target sample rate of the audio. Defaults to 32000.
        n_channels (int, optional): The number of channels in the audio. Defaults to 1.
        cutoff (str, optional): The method to handle segments that are shorter than segment_length_s. 
                               Can be 'pad', 'leave', or 'crop'. Defaults to 'pad'.
        overlap (float, optional): The overlap between consecutive segments as a fraction of segment_length_s. 
                                   Defaults to 0.0.
        normalize (bool, optional): Whether to normalize the audio. Defaults to False.
        denoise (bool, optional): Whether to denoise the audio. Defaults to False.
        desilence (bool, optional): Whether to remove silence from the audio. Defaults to False.

    Returns:
        None
    """

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
