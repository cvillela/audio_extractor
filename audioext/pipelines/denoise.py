from tqdm import tqdm
import numpy as np
import os
import argparse
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
    ndarray32_to_audiosegment,
    bandpass_filter_signal,
    get_freq_domain,
    normalize_unit,
)

from ..audio.audio_utils import (
    list_wavs_from_dir,
    plot_bp_filter,
    plot_spectrogram,
    plot_time,
    plot_fft,
)

from ..constants import constants


def main(args):
    if args.output_dir is None:
        output_dir = os.path.join(args.samples_dir, "processed/")
    else:
        output_dir = args.output_dir

    # Check if out directory exists, if not, create it
    os.makedirs(output_dir, exist_ok=True)
    
    if args.remove_speech:
        voice_detection_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                        use_auth_token=constants.HF_AUTH_TOKEN)

    file_paths = list_wavs_from_dir(args.samples_dir, walk=False)

    for f in tqdm(file_paths):
        print(f"Processing {f}")
        
        audio = AudioSegment.from_file(f)
        audio = audio.set_channels(1)
        sr = audio.frame_rate

        # remove speech
        if args.remove_speech:
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
            
        
        ### Convert to float32
        y = audiosegment_to_ndarray_32(audio)

        # denoise -> 2 passes is more effective
        y_red = nr.reduce_noise(y=y, sr=sr, stationary=True)
        y_red = nr.reduce_noise(y=y_red, sr=sr, stationary=False)

        y_final = y_red
        if args.band_pass:
            # nyquist
            high = args.high
            if high >= sr / 2:
                high = (sr / 2) - 10

            # bandpass
            y_bp = bandpass_filter_signal(
                y_red, sr, order=6, low=args.low, high=high, plot=False
            )
            y_final = y_bp

        # remove silence
        if args.remove_silence:
            audio_nonsilent = 0
            curr_thresh = args.silence_thresh

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
        wavfile.write(os.path.join(output_dir, filename), rate=sr, data=y_final)

        if args.plot:
            Y_db = get_freq_domain(y)
            Y_red = get_freq_domain(y_red)
            Y_final = get_freq_domain(y_final)

            plot_spectrogram(Y=Y_db, sr=sr, title="Original", y_axis="log", save=True)
            plot_spectrogram(
                Y=Y_red, sr=sr, title="Reduce Noise", y_axis="log", save=True
            )

            if args.band_pass:
                Y_bp = get_freq_domain(y_bp)
                plot_spectrogram(
                    Y=Y_bp, sr=sr, title="Bandpass Filter", y_axis="log", save=True
                )
            plot_spectrogram(
                Y=Y_final,
                sr=sr,
                title="Final - Remove Silence",
                y_axis="log",
                save=True,
            )
            print(f"{len(y_final)/len(y):.0%} of original")


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

    # parser.add_argument(
    #     "--start",
    #     type=float,
    #     default=0,
    #     help="Start (in seconds) of the segment. Default is 0.00s",
    # )

    parser.add_argument(
        "--remove_speech",
        default=True,
        help="Wether to remove speech. Default is true",
        action=argparse.BooleanOptionalAction,
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

    parser.add_argument(
        "--plot",
        default=False,
        help="Wether to plot data. Default is false (--no-plot)",
        action=argparse.BooleanOptionalAction,
    )

    # Parse the arguments
    args = parser.parse_args()

    main(args)
