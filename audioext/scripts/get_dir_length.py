from ..audio.audio_utils import list_wavs_from_dir, get_len_wavs
import argparse

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Parsing Arguments")
    parser.add_argument(
        "--samples_dir", type=str, default="", help="Path to directory wav samples."
    )
    args = parser.parse_args()

    wavs = list_wavs_from_dir(args.samples_dir)
    dur = get_len_wavs(wavs)

    print(
        f"Directory {args.samples_dir} contains {len(wavs)} files and has a duration of {dur} hours."
    )
