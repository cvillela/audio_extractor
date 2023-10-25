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
from multiprocessing import Pool

import noisereduce as nr
from pydub.effects import compress_dynamic_range
from scipy.io import wavfile
from ..audio.audio_processer import (
    normalize_loudness,
    audiosegment_to_ndarray_32,
    ndarray32_to_audiosegment,
    bandpass_filter_signal,
    get_freq_domain,
    normalize_unit,
    get_silence_ranges,
    split_on_silence
)

from ..audio.audio_utils import (
    list_wavs_from_dir,
    plot_bp_filter,
    plot_spectrogram,
    plot_time,
    plot_fft,
)

from ..constants import constants

def denoise_wrapper(args_tuple):
    # Unpack the tuple
    f, args, se = args_tuple
    return denoise_single(f, args, se)


def get_speech_markers(f, voice_detection_pipeline):      
        
        output = voice_detection_pipeline(f)        
        se = []
        for speech in output.get_timeline().support():
            se.append([speech.start*1000, speech.end*1000])
        return se
    

def denoise_single(f, args, se):
    print(f"Processing {f}")   
    
    try:    
        audio = AudioSegment.from_file(f)
        audio = audio.set_channels(1)
        audio = normalize_loudness(audio)
        audio = compress_dynamic_range(audio)
    except Exception as e:
        print(f"Found error {e} in file {f}")
        with open("errors.txt", "a") as err_file:
            err_file.write(f"file {f}-> {str(e)}" + os.linesep)
        return
    
    # remove speech
    if args.remove_speech:
        start_pos = 0
        no_speech = AudioSegment.silent(duration=0)    
        for start, end in se:
            no_speech = no_speech+audio[start_pos:start]
            start_pos = end
            
        no_speech = no_speech+audio[start_pos:]
        audio = no_speech
        
    # np audio
    sr = audio.frame_rate
    y = audiosegment_to_ndarray_32(audio)

    if args.denoise:    
        # denoise -> 2 passes is more effective
        y_red = nr.reduce_noise(y=y, sr=sr, stationary=True)
        y_red = nr.reduce_noise(y=y_red, sr=sr, stationary=False)
        y = y_red
        
    if args.band_pass:
        # nyquist
        high = args.high
        if high >= sr / 2:
            high = (sr / 2) - 10

        # bandpass
        y_bp = bandpass_filter_signal(
            y, sr, order=6, low=args.low, high=high, plot=False
        )
        y = y_bp

    # remove silence
    if args.remove_silence:
        y = normalize_unit(y)
        audio_prep = ndarray32_to_audiosegment(y, frame_rate=sr)
        audio_prep = normalize_loudness(audio_prep)
        audio_prep = compress_dynamic_range(audio_prep)
        
        silence_ranges = []
        curr_thresh = args.silence_thresh
        count = 0
    
        while silence_ranges == [] and count <= 2:
            silence_ranges = get_silence_ranges(
                audio_prep,
                min_silence_len=args.min_silence_len,
                silence_thresh=curr_thresh,
                keep_silence=args.keep_silence,
                seek_step=args.seek_step,
            )
            curr_thresh -= 10
            count += 1
            
        if silence_ranges == []:
            print(f"Cant remove silence in file {f}")
            audio_nonsilent = audio_prep
            with open("errors.txt", "a") as err_file:
                err_file.write(f"file {f}-> remove silence failed." + os.linesep)
        else:
            audio_nonsilent = split_on_silence(
                audio_prep,
                silence_ranges,
            )
            audio_nonsilent = normalize_loudness(sum(audio_nonsilent))
            y = audiosegment_to_ndarray_32(audio_nonsilent)
            
            # save original cropped
            if args.save_no_speech:
                original_crop = split_on_silence(
                    audio,
                    silence_ranges,
                )
                original_crop = normalize_loudness(sum(original_crop))
                y_crop = audiosegment_to_ndarray_32(original_crop)        
            
    # normalize unit and export
    y = normalize_unit(y)
    filename = f.split("/")[-1]
    wavfile.write(os.path.join(args.output_dir, filename), rate=sr, data=y)
    
    if args.save_no_speech:
        y_crop = normalize_unit(y_crop)
        filename = f.split("/")[-1]
        wavfile.write(os.path.join(args.no_speech_output_dir, filename), rate=sr, data=y_crop)
    
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
        help="Path to directory to save the denoised audio to. Defaults to samples_dir/processed.",
    )
    parser.add_argument(
        "--no_speech_output_dir",
        type=str,
        help="Path to directory to save the no speech audio to. Defaults to samples_dir/no_speech.",
    )

    parser.add_argument(
        "--n_processes",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of processes to use. Defaults to the number of available CPUs.",
    )
    
    parser.add_argument(
        "--save_no_speech",
        default=False,
        help="Wether to save NO SPEECH audio. Defaults to false",
        action=argparse.BooleanOptionalAction,
    )
    
    # remove_speech
    parser.add_argument(
        "--remove_speech",
        default=True,
        help="Wether to remove speech. Default is true",
        action=argparse.BooleanOptionalAction,
    )

    # Denoise args
    parser.add_argument(
        "--denoise",
        default=True,
        help="Denoise signal from background noise. Default is true",
        action=argparse.BooleanOptionalAction,
    )

    # Band pass arguments
    parser.add_argument(
        "--band_pass",
        default=True,
        help="Band-Pass signal after denoising. Default is true",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--low",
        type=int,
        default=512,
        help="Filter frequencies below. Default is 512Hz",
    )
    parser.add_argument(
        "--high",
        type=int,
        default=10000,
        help="Filter frequencies above. Default is 11000Hz",
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
    
    print(f"There are {args.n_processes} processes.")
    start = time()
    
    if args.output_dir is None:
        output_dir = os.path.join(args.samples_dir, "processed/")
        args.output_dir = output_dir
    else:
        output_dir = args.output_dir
    
    # Check if out directory exists, if not, create it
    os.makedirs(output_dir, exist_ok=True)
        
    if args.save_no_speech:
        if args.no_speech_output_dir is None:
            no_speech_output_dir = os.path.join(args.samples_dir, "no_speech/")
            args.no_speech_output_dir = no_speech_output_dir
        else:
            no_speech_output_dir = args.no_speech_output_dir
        # Check if out directory exists, if not, create it
        os.makedirs(no_speech_output_dir, exist_ok=True)

    file_paths = list_wavs_from_dir(args.samples_dir, walk=False)

    if args.remove_speech:
        
        print(f"Getting speech markers from {len(file_paths)} files!")
        voice_detection_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                        use_auth_token=constants.HF_AUTH_TOKEN)

        speech_markers = []
        for f in tqdm(file_paths):
            speech_markers.append(get_speech_markers(f, voice_detection_pipeline))
        print("Done!")
    else:
        speech_markers = [[] for _ in file_paths]
        
    # Create a list of argument tuples
    args_list = [(f, args, se) for f, se in zip(file_paths, speech_markers)]

    with multiprocessing.Pool(processes=args.n_processes) as pool:
        results = pool.map(denoise_wrapper, args_list)
    
    end = time()
    print(f"Multiprocess took {end-start} seconds")
    
    
# Instantiate voice activity detection model
# vad_model = Model.from_pretrained("pyannote/segmentation-3.0", 
#                     use_auth_token=constants.HF_AUTH_TOKEN)
# voice_detection_pipeline = VoiceActivityDetection(segmentation=vad_model)
# HYPER_PARAMETERS = {
#     # remove speech regions shorter than that many seconds.
#     "min_duration_on": 0.0,
#     # fill non-speech regions shorter than that many seconds.
#     "min_duration_off": 0.0
# }
# voice_detection_pipeline.instantiate(HYPER_PARAMETERS)
        