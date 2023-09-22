from tqdm import tqdm
import numpy as np
import os
import argparse
from tqdm import tqdm
from time import time
from pydub import AudioSegment

import noisereduce as nr
from pydub.silence import split_on_silence
from scipy.io import wavfile
from ..audio.audio_processer import (
    normalize_loudness,
    audiosegment_to_ndarray_32,
    ndarray32_to_audiosegment,
    bandpass_filter_signal,
    get_freq_domain,
)

from ..audio.audio_utils import (
    list_wavs_from_dir,
    plot_bp_filter,
    plot_spectrogram,
    plot_time,
    plot_fft,
)


def main(args):
    if args.output_dir is None:
        output_dir = os.path.join(args.sample_dir, "processed/")

    # Check if out directory exists, if not, create it
    os.makedirs(output_dir, exist_ok=True)

    file_paths = list_wavs_from_dir(args.samples_dir)
    for f in tqdm(file_paths):
        start = time()

        s1 = time()
        # load sample
        audio = AudioSegment.from_file(f)
        audio = audio.set_channels(1)
        sr = audio.frame_rate
        # print(f"Load :{time()-s1} secs")

        s1 = time()
        ### Normalize and Convert to float32
        norm_audio = normalize_loudness(audio)
        y = audiosegment_to_ndarray_32(norm_audio)
        # print(f"Normalize + Convert : {time()-s1} secs")

        s1 = time()
        # denoise
        y_red = nr.reduce_noise(y=y, sr=sr, stationary=args.stationary)
        # print(f"Denoise : {time()-s1} secs")

        y_final = y_red
        s1 = time()
        if args.band_pass:
            # bandpass
            y_bp = bandpass_filter_signal(
                y_red, sr, order=6, low=args.low, high=args.high, plot_filter=False
            )
            y_final = y_bp
        # print(f"Bandpass : {time()-s1} secs")
        
        s1 = time()
        # remove silence
        if args.remove_silence:
            audio_denoised = ndarray32_to_audiosegment(y_final, frame_rate=sr)

            audio_chunks = split_on_silence(
                audio_denoised,
                min_silence_len=args.min_silence_len,
                silence_thresh=args.silence_thresh,
                keep_silence=args.keep_silence,
                seek_step=args.seek_step,
            )

            audio_nonsilent = sum(audio_chunks)
            # print(f"Split on Silence {time()-s1} secs")

            s1 = time()
            y_final = audiosegment_to_ndarray_32(audio_nonsilent)
            # print(f"Convert back : {time()-s1} secs")

        # print(f"Pipe : {time() - start} seconds.")
        # print(f"Original audio is {len(audio)/1000} seconds long.")
        # print(f"Final audio is {len(audio_nonsilent)/1000} seconds long.")

        if args.plot:
            start = time()

            Y_db = get_freq_domain(y)
            Y_red = get_freq_domain(y_red)
            Y_bp = get_freq_domain(y_bp)
            Y_final = get_freq_domain(y_final)

            plot_spectrogram(Y=Y_db, sr=sr, title="Original", y_axis="log", save=True)
            plot_spectrogram(
                Y=Y_red, sr=sr, title="Reduce Noise", y_axis="log", save=True
            )
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
            plot_time(y, sr=sr, title="Original")
            plot_time(y_final, sr=sr, title="Final")
            # print(f"Plots : {time()-start} seconds.")

        filename = f.split("/")[-1]
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
    # Denoise args
    parser.add_argument(
        "--stationary",
        default=False,
        help="Stationary reduce noise. Default is False",
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
        "--silence_tresh",
        type=int,
        default=-30,
        help="Values below this dB are considered silence. Default is -30dB",
    )
    parser.add_argument(
        "--seek_step",
        type=int,
        default=100,
        help="Seek step for segment on silence in ms. Default is 100ms.",
    )
    parser.add_argument(
        "--keep_silence",
        type=int,
        default=500,
        help="Silence to keep between segments. Default is 500ms.",
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
