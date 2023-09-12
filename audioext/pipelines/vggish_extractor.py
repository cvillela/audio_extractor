from tqdm import tqdm
import numpy as np
import os
import argparse
import torch

from ..audio.audio_segmenter import segment_audio
from ..audio.audio_utils import list_wavs_from_dir


def extract_from_files(file_paths, out_dir, mult_factor=100, emb_chunk_size=1000, **segment_kwargs):
    """
    Extracts audio samples from a list of file paths, processes them in batches, and saves the extracted embeddings to disk.

    Parameters:
    - file_paths (list): A list of file paths to the audio files.
    - out_dir (str): The directory to save the extracted embeddings.
    - batch_size (int, optional): The number of audio samples to process in each batch. Defaults to 4.
    - meanpool (bool, optional): Whether to use mean pooling to calculate embeddings. Defaults to False.
    - mult_factor (int, optional): Number of embeddings to be extracted from each audio segment. Defaults to 100.
    - emb_chunk_size (int, optional): The maximum number of embeddings to save in each chunk. Defaults to 1000.
    - **segment_kwargs: Additional keyword arguments to be passed to the segment_audio function.

    Returns:
    - None
    """

    print(f"Extracting embeddings to {out_dir}.")
    
    model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    model.eval()
    
    
    i = 0
    emb_list = []    
    
    for f in tqdm(file_paths):
        curr_samples, curr_meta = segment_audio(f, **segment_kwargs)
        for s, meta in zip(curr_samples, curr_meta):
            s = s.astype(np.int16)
            emb = model.forward(s, fs=meta["sample_rate"]).cpu().detach().numpy()
            emb_list.append(emb)
            
        if len(emb_list) > emb_chunk_size:
            i+=1
            emb_list = np.vstack(emb_list)
            np.save(os.path.join(out_dir, f"m{mult_factor}_{i}.npy"), emb_list)
            emb_list = []

    
    # process last batch -> sample_list with < batch_size elements or emb_list with < emb_chunk_size elements
    if len(emb_list)>0:                
                
        emb_list = np.vstack(emb_list)
        i+=1
        np.save(os.path.join(out_dir, f"m{mult_factor}_{i}.npy"), emb_list)

    return


def main(args):
        
    # Check if out directory exists, if not, create it
    os.makedirs(args.output_dir, exist_ok=True)

    # specify segmentation parameters     
    seg_dict = {
        "segment_length_s": args.seg_len,
        "cutoff": args.cutoff,
        "overlap": args.overlap,
        "target_sr": None,
        "n_channels": 1,
        "loudness_norm": True,
        "normalize_amplitude": True
    }
    
    # get file paths from sample dir
    file_paths = list_wavs_from_dir(args.samples_dir)

    extract_from_files(
        file_paths, args.output_dir, emb_chunk_size=args.chunk_size, **seg_dict
    )


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Parsing Arguments")

    # Add arguments
    parser.add_argument("--samples_dir", type=str, help="Path to directory containing audio samples.")
    parser.add_argument("--output_dir", type=str, help="Path to directory to save the embeddings to.")
    
    parser.add_argument("--chunk_size", type=int, default=1000, help="Number of embedding chunks to save in each file for saving RAM. Defaults to 1000.")
    
    parser.add_argument("--seg_len", default=5, type=int, help="Duration of audio segments in seconds. Default is 5, maximum is 23.")
    parser.add_argument("--cutoff", type=str, default="crop",  help="Wether to ignore generated samples with length < seg_len, or pad them with silence. Can be ['crop', 'pad']. Default is crop")
    parser.add_argument("--overlap", default=0.0, help="Percentage of overlap between samples. Default is 0.00.")

    # parser.add_argument("--verbose", default=False, help="Wether to print logs. Default is false (--no-verbose)", action=argparse.BooleanOptionalAction)

    # Parse the arguments
    args = parser.parse_args()

    main(args)
