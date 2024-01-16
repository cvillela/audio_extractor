from tqdm import tqdm
import numpy as np
import os
import argparse
from tqdm import tqdm
from time import time
from pydub import AudioSegment

from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Pipeline

import multiprocessing

from ..audio.audio_utils import (
    list_wavs_from_dir,
)

from ..constants import constants

def merge_intervals(intervals):
    if not intervals:
        return []

    # Combine and sort intervals
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]
    for current in intervals:
        previous = merged[-1]

        if current[0] <= previous[1]:
            # Merge intervals if they overlap
            merged[-1] = (previous[0], max(previous[1], current[1]))
        else:
            # Add non-overlapping interval to the list
            merged.append(current)

    return merged


def denoise_single(f, args, vad_pipeline):
    print(f"Processing {f}")   
    
    try:    
        audio = AudioSegment.from_file(f)
    except Exception as e:
        print(f"Found error {e} in file {f}")
        with open("errors.txt", "a") as err_file:
            err_file.write(f"file {f}-> {str(e)}" + os.linesep)
        return
    
    ini_len = len(audio)

    # Get Speech intervals from each channel
    channel_intervals = []
    mono_files = audio.split_to_mono()
    for mono in mono_files:
        mono.set_frame_rate(32000)
        mono.export("temp.wav", format="wav")
        output = vad_pipeline("temp.wav")
        for speech in output.get_timeline().support():
            channel_intervals.append([speech.start*1000, speech.end*1000])

    # Merge intervals from all channels
    merged_intervals = merge_intervals(channel_intervals)

    # Remove speech
    start_pos = 0
    no_speech = AudioSegment.silent(duration=0)
    for start, end in merged_intervals:
        no_speech = no_speech+audio[start_pos:start]
        start_pos = end
    no_speech = no_speech+audio[start_pos:]
    audio = no_speech
    
    
    final_len = len(audio)

    filename = f.split("/")[-1]
    audio.export(os.path.join(args.no_speech_output_dir, filename), format="wav")
    print(f"Removed {int((final_len-ini_len)/1000)} seconds of speech from {filename}.")
    
if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Parsing Arguments")

    # Add arguments
    parser.add_argument(
        "--samples_dir", type=str, help="Path to directory containing audio samples."
    )
    parser.add_argument(
        "--no_speech_output_dir",
        type=str,
        help="Path to directory to save the no speech audio to. Defaults to samples_dir/no_speech.",
    )


    # Parse the arguments
    args = parser.parse_args()    
    
    if args.no_speech_output_dir is None:
        no_speech_output_dir = os.path.join(args.samples_dir, "no_speech/")
        args.no_speech_output_dir = no_speech_output_dir
    else:
        no_speech_output_dir = args.no_speech_output_dir
    # Check if out directory exists, if not, create it
    os.makedirs(no_speech_output_dir, exist_ok=True)

    file_paths = list_wavs_from_dir(args.samples_dir, walk=False)
    
    start_speech = time()
    print("Starting Speech Removal!")
    # Instantiate voice activity detection model
    vad_model = Model.from_pretrained("pyannote/segmentation-3.0", 
                        use_auth_token=constants.HF_AUTH_TOKEN)
    voice_detection_pipeline = VoiceActivityDetection(segmentation=vad_model)
    HYPER_PARAMETERS = {
        # remove speech regions shorter than that many seconds.
        "min_duration_on": 0.0,
        # fill non-speech regions shorter than that many seconds.
        "min_duration_off": 0.0
    }
    voice_detection_pipeline.instantiate(HYPER_PARAMETERS)

    for f in tqdm(file_paths):
        denoise_single(f, args, vad_pipeline=voice_detection_pipeline)
    
    end_speech = time()
    print(f"Multiprocess took {end_speech-start_speech} seconds")
        