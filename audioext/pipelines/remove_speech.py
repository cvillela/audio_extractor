from tqdm import tqdm
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from time import time
from pydub import AudioSegment
from pyannote.audio import Pipeline

import noisereduce as nr
from pydub.silence import split_on_silence
from pydub.effects import compress_dynamic_range
from scipy.io import wavfile
from ..audio.audio_processer import (
    normalize_loudness,
    audiosegment_to_ndarray_32,
    normalize_unit,
    ndarray32_to_audiosegment,
)

from ..audio.audio_utils import (
    list_wavs_from_dir,
)

from ..constants import constants


def main(args):
    if args.output_dir is None:
        output_dir = os.path.join(args.samples_dir, "no_speech/")
    else:
        output_dir = args.output_dir

    # Check if out directory exists, if not, create it
    os.makedirs(output_dir, exist_ok=True)
    
    voice_detection_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                    use_auth_token=constants.HF_AUTH_TOKEN)

    file_paths = list_wavs_from_dir(args.samples_dir, walk=False)
    
    for f in tqdm(file_paths):
        print(f"Processing {f}")
        
        audio = AudioSegment.from_file(f)
        # audio = audio.set_channels(1)
        sr = audio.frame_rate

        output = voice_detection_pipeline(f)
        se = []
        for speech in output.get_timeline().support():
            se.append([speech.start*1000, speech.end*1000])
        start_pos = 0
        no_speech = AudioSegment.silent(duration=0)    
        for start, end in se:
            no_speech = no_speech+audio[start_pos:start]
            start_pos = end

        no_speech = no_speech+audio[start_pos:]
        audio = no_speech
        
        # remove silence
        if args.remove_silence:
            audio_nonsilent = 0
            curr_thresh = args.silence_thresh
            y_final = audiosegment_to_ndarray_32(audio)
            y_final = normalize_unit(y_final)
            audio_denoised = ndarray32_to_audiosegment(y_final, frame_rate=sr)
            audio_denoised = normalize_loudness(audio_denoised)
            audio_denoised = compress_dynamic_range(audio_denoised)

            while audio_nonsilent == 0:
                audio_chunks = split_on_silence(
                    audio_denoised,
                    min_silence_len=args.min_silence_len,
                    silence_thresh=curr_thresh,
                    keep_silence=args.keep_silence,
                    seek_step=args.seek_step,
                )
                audio_nonsilent = sum(audio_chunks)
                curr_thresh -= 10
            audio_nonsilent = normalize_loudness(audio_nonsilent)
            y_final = audiosegment_to_ndarray_32(audio_nonsilent)

        # normalize unit and export
        y_final = normalize_unit(y_final)

        filename = f.split("/")[-1]
        filename = os.path.basename(f)

        
        wavfile.write(os.path.join(output_dir, filename), rate=sr, data=y_final)


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Parsing Arguments")

    # Add arguments
    parser.add_argument(
        "--samples_dir", type=str, help="Path to directory containing audio samples."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory to save the embeddings to. Defaults to samples_dir/processed.",
    )
   
    # Segment on Silence args
    parser.add_argument(
        "--remove_silence",
        default=True,
        help="Wether to remove silence. Default is true",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--min_silence_len",
        type=int,
        default=2000,
        help="Minimal length to be considered silence. Default is 2000ms",
    )
    parser.add_argument(
        "--silence_thresh",
        type=int,
        default=-45,
        help="Values below this dB are considered silence. Default is -30dB",
    )
    parser.add_argument(
        "--keep_silence",
        type=int,
        default=200,
        help="Silence to keep between segments. Default is 500ms.",
    )
    parser.add_argument(
        "--seek_step",
        type=int,
        default=100,
        help="Seek step for segment on silence in ms. Default is 100ms.",
    )

    # Parse the arguments
    args = parser.parse_args()

    main(args)
