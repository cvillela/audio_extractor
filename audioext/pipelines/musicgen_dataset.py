from tqdm import tqdm
import numpy as np
import os
import argparse

from ..audio.audio_segmenter import segment_audio, save_sample_meta, generate_splits
from ..audio.audio_utils import list_wavs_from_dir, get_len_wavs


JSON_TEMPLATE = {
    "key": "",
    "artist": "",
    "sample_rate": None,
    "file_extension": "wav",
    "description": "",
    "keywords": "",
    "duration": 30.0,
    "bpm": "",
    "genre": "",
    "title": "",
    "name": "", #important!!!! matches wav file name
    "instrument": "",
    "moods": []
}



def main(args):

    # Check if out directory exists, if not, create it
    os.makedirs(args.output_dir, exist_ok=True)
    
    # specify segmentation parameters     
    seg_dict = {
        "segment_length_s": args.seg_len,
        "cutoff": args.cutoff,
        "overlap": args.overlap,
        "target_sr": 32000,
        "n_channels": 1,
        "loudness_norm": True,
        "normalize_amplitude": True
    }
    
    # get file paths from sample dir
    file_paths = list_wavs_from_dir(args.samples_dir)
    
    # add random prompt for conditioning
    random_prompt = "8H38fNdtri"
    
    print("Extracting samples....")
    print(f"Prompt to be used in music generation/conditioning is: {random_prompt}")
    
    for f in tqdm(file_paths):
        seg_list, meta_list = segment_audio(
            f, **seg_dict
        )
        for segment, meta in zip(seg_list, meta_list):
            meta_musicgen = JSON_TEMPLATE.copy()
            meta_musicgen["sample_rate"] = meta["sample_rate"]
            meta_musicgen["description"] = random_prompt
            meta_musicgen["keywords"] = random_prompt
            meta_musicgen["genre"] = "classical, orchestra"
            
            save_sample_meta(meta_musicgen, segment, args.output_dir)
    
    print("Segmentation is done!")
    sample_wavs = list_wavs_from_dir(args.output_dir)
    print("There are {} samples in total".format(len(sample_wavs)))
    print("Duration of dataset is {} hours".format(get_len_wavs(sample_wavs)))
    
    print("Creating splits...")
    generate_splits(args.output_dir, val_ratio=args.val_ratio, n_test=args.n_test)
    print("Done!")
    
if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Parsing Arguments")

    # Add arguments
    parser.add_argument("--samples_dir", type=str, default="", help="Path to directory containing raw samples.")
    parser.add_argument("--output_dir", type=str, default="", help="Path to directory to save the cropped samples and metadata to.")
    
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Percentage of files to be used for validation. Default is 0.2 (20%)")
    parser.add_argument("--n_test", type=int, default=50, help="Number of files to be used for testing/generation. Default is 50")
    
    parser.add_argument("--seg_len", default=30, type=int, help="Duration of audio segments in seconds. Default is 30.")
    parser.add_argument("--cutoff", type=str, default="crop",  help="Wether to ignore generated samples with length < seg_len, or pad them with silence. Can be ['crop', 'pad']. Default is crop")
    parser.add_argument("--overlap", default=0.0, help="Percentage of overlap between samples. Default is 0.00.")

    parser.add_argument("--verbose", default=False, help="Wether to print logs. Default is false (--no-verbose)", action=argparse.BooleanOptionalAction)

    # Parse the arguments
    args = parser.parse_args()

    main(args)
