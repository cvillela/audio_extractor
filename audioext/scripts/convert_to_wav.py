
from ..audio.audio_utils import list_wavs_from_dir, get_len_wavs
from pydub import AudioSegment
import argparse
import os

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Parsing Arguments")
    parser.add_argument(
        "--samples_dir", type=str, default="", help="Path to directory wav samples."
    )
    args = parser.parse_args()


    file_paths = list_wavs_from_dir(args.samples_dir, walk=False)
    for f in file_paths:
        if not f.endswith(".wav"):
            print(f)
            wav_filename = f[:-4] + ".wav"
            sound = AudioSegment.from_file(f)
            sound.export(wav_filename, format='wav')
            os.remove(f)