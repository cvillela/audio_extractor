import os
import argparse

from ..audio.audio_utils import list_wavs_from_dir
from ..models import jukebox
from ..models import mn


def main(args):
    if args.output_dir is None:
        args.output_dir = os.path.join(args.samples_dir, "embeddings")

    # Check if out directory exists, if not, create it
    os.makedirs(args.output_dir, exist_ok=True)

    if args.full_track:
        args.meanpool = True
        args.emb_chunk_size = 50000
        args.batch_size = 1
        args.cutoff = "pad"

    # specify segmentation parameters
    seg_dict = {
        "segment_length_s": args.seg_len,
        "cutoff": args.cutoff,
        "overlap": args.overlap,
        "target_sr": 0,
        "n_channels": 1,
        "loudness_norm": True,
        "normalize_amplitude": True,
    }

    # get file paths from sample dir
    file_paths = list_wavs_from_dir(args.samples_dir, walk=False)

    match (args.model):
        case "jukebox":
            # Check if segment length is in expected range
            assert (
                args.seg_len <= jukebox.MAX_SEG_LEN_S
            ), f"Segment length is too long! Must be {jukebox.MAX_SEG_LEN_S} seconds or less."

            seg_dict["target_sr"] = jukebox.JUKEBOX_SR
            if args.full_track:
                jukebox.extract_full_track(
                    file_paths,
                    args.output_dir,
                    emb_chunk_size=args.chunk_size,
                    **seg_dict,
                )
            else:
                jukebox.extract_segments(
                    file_paths,
                    args.output_dir,
                    batch_size=args.batch_size,
                    meanpool=args.meanpool,
                    mult_factor=args.mult_factor,
                    emb_chunk_size=args.chunk_size,
                    **seg_dict,
                )
        case "mn":
            seg_dict["target_sr"] = 32000
            mn.extract_full_track(
                file_paths, args.output_dir, emb_chunk_size=args.chunk_size, **seg_dict
            )


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Parsing Arguments")

    # Add arguments
    parser.add_argument(
        "--model", type=str, default="jukebox", help="Model Name ['jukebox, mn']."
    )

    parser.add_argument(
        "--samples_dir", type=str, help="Path to directory containing audio samples."
    )
    parser.add_argument(
        "--output_dir", type=str, help="Path to directory to save the embeddings to."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size, defaults to 4."
    )
    parser.add_argument(
        "--meanpool",
        default=False,
        help="Wether to perform mean pooling. Default is false (--no-meanpool)",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--full_track",
        default=False,
        help="Wether to extract a single embedding for each track. Default is false.",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--mult_factor",
        type=int,
        default=100,
        help="Number of embeddings to be extracted from each audio segment. Defaults to 100. Meanpooling overrides this argument",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100000,
        help="Number of embedding chunks to save in each file for saving RAM. Defaults to 100000.",
    )
    parser.add_argument(
        "--seg_len",
        default=5,
        type=int,
        help="Duration of audio segments in seconds. Default is 5, max depends on the model.",
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default="leave",
        help="Wether to ignore generated samples with length < seg_len, or pad them with silence. Can be ['crop', 'pad', 'leave']. Default is leave",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Percentage of overlap between samples. Default is 0.00.",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        help="Wether to print logs. Default is false (--no-verbose)",
        action=argparse.BooleanOptionalAction,
    )

    # Parse the arguments
    args = parser.parse_args()

    main(args)
