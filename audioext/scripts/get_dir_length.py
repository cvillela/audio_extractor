# from ..audio.audio_utils import list_wavs_from_dir, get_len_wavs
import argparse
import os
from pydub import AudioSegment


def list_wavs_from_dir(path, walk=True):
    file_paths = []

    if walk:
        for dirpath, _, filenames in os.walk(path):
            for file_name in filenames:
                if (
                    file_name.lower().endswith(".wav")
                    or file_name.lower().endswith(".mp3")
                    or file_name.lower().endswith(".flac")
                    or file_name.lower().endswith(".m4a")
                ):
                    file_paths.append(os.path.join(dirpath, file_name))
    else:
        for item in os.listdir(path):
            if os.path.isfile(os.path.join(path, item)) and (
                item.lower().endswith(".wav")
                or item.lower().endswith(".mp3")
                or item.lower().endswith(".flac")
                or item.lower().endswith(".m4a")
            ):
                file_paths.append(os.path.join(path, item))

    return file_paths


def get_len_wavs(file_paths):
    dur = 0
    for f in file_paths:
        audio = AudioSegment.from_file(f)
        dur += len(audio)

    return dur / (1000 * 60 * 60)  # in hours


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Parsing Arguments")
    parser.add_argument(
        "--samples_dir", type=str, default="", help="Path to directory wav samples."
    )
    parser.add_argument(
        "--walk",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    wavs = list_wavs_from_dir(args.samples_dir, walk=args.walk)
    dur = get_len_wavs(wavs)

    print(
        f"Directory {args.samples_dir} contains {len(wavs)} files and has a duration of {dur} hours."
    )
