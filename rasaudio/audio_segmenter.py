
import os
from pydub import AudioSegment, silence, effects
import uuid
from unidecode import unidecode
import json
import numpy as np
import scipy.io.wavfile

from audio_processer import normalize_unit, audiosegment_to_ndarray_32

def get_audio_metadata(audio, filename):
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
        "filename":filename
    }

    return meta
        
def segment_audio(
        file_path, out_dir, segment_length_s=10, target_sr=32000, n_channels=1,
        cutoff='pad', overlap=0.0, normalize_loudness=True, normalize_unit=True
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
    
    # normalize loudness
    if normalize_loudness:
        audio = effects.normalize(audio)
    
    # Audio info dict
    audio_metadata = get_audio_metadata(audio, file_name)
    
    # turn into np array 
    np_audio = audiosegment_to_ndarray_32(audio)
    
    # normalize between -1 and 1
    if normalize_unit:
        np_audio = normalize_unit(np_audio)

    # segment into segment_length_s samples
    segment_length_samples = segment_length_s * audio.frame_rate
    step = int((1-overlap)*segment_length_samples)
    for i in range(0, len(np_audio), step):
        # create segment
        start_time = i 
        end_time = i + segment_length_samples
        segment = np_audio[start_time:end_time][:,0]
        
        # pad or crop end-of-file samples
        if end_time > len(audio):
            if cutoff=='pad': # pad with silence (0 amplitude)
                # if len(segment) < 0.5*segment_length_samples: # pad at most 50% of the signal
                #     break
                pad_len = segment_length_samples - len(segment)
        
                # Pad with 0s
                pad = np.zeros(pad_len)
                segment = np.concatenate((segment,pad), axis=0)
            
            elif cutoff=='leave': # save sample smaller than segment_length_s
                segment = audio[start_time:len(audio)] 
            
            elif cutoff=='crop': # discard smaller sample
                break
            
        save_sample_meta(audio_metadata, segment, out_dir)

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
    scipy.io.wavfile.write(seg_path, segment_meta_dict["sample_rate"], segment)
    
    # Save the metadata as a JSON file
    with open(meta_path, 'w', encoding='utf8') as fp:
        json.dump(segment_meta_dict, fp)
