import jukemirlib
from tqdm import tqdm
import numpy as np
import os
import argparse

from ..audio.audio_segmenter import segment_audio
from ..audio.audio_utils import list_wavs_from_dir

JUKEBOX_SR = 44100
CTX_WINDOW_LENGTH = 1048576

def extract_batch(audio_samples, meanpool=False, mult_factor=100):
    """
    Extracts embeddings from a batch of audio samples using the Jukebox library.

    Parameters:
        audio_samples (ndarray): An array of audio samples.
        meanpool (bool, optional): Whether to apply mean pooling to the embeddings. Defaults to False.
        mult_factor (int, optional): The factor by which to split the embeddings. Must be less than or equal to 1722. Defaults to 100.

    Returns:
        ndarray: The final embeddings after extracting and processing the audio samples.
    """
    assert mult_factor <= 1722
        
    embs = jukemirlib.extract(audio_samples, layers=[36], meanpool=meanpool)[36]

    # if batch_size == 1
    if embs.ndim==2:
        embs = embs.reshape(1, embs.shape[0], embs.shape[1])
        
    if meanpool:
        print("Applying mean pooling to embeddings, mult factor is disconsidered")
        final_embs=embs
    else:
        split_embeddings = np.array_split(embs, mult_factor, axis=1)
        mean_splits = [np.mean(arr, axis=1) for arr in split_embeddings]
        final_embs = np.vstack(mean_splits)
    
    return final_embs



def extract_from_files(file_paths, out_dir, batch_size=4, meanpool=False, mult_factor=100, emb_chunk_size=1000, **segment_kwargs):
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

    print(f"Extracting embeddings to {out_dir} with a batch size of {batch_size} and m_factor of {mult_factor}")
    
    i = 0
    emb_list = []
    sample_list = []
    curr_batch = []
    
    for f in tqdm(file_paths):
        sample_list, _ = segment_audio(f, **segment_kwargs)

        if len(sample_list) >= batch_size:
            while len(sample_list) >= batch_size:            
                for _ in range(batch_size):
                    curr_batch.append(sample_list.pop())
                emb_list.append(extract_batch(curr_batch, meanpool=meanpool, mult_factor=mult_factor))
                curr_batch = []
        
        if len(emb_list) > emb_chunk_size:
            i+=1
            emb_list = np.vstack(emb_list)
            np.save(os.path.join(out_dir, f"m{mult_factor}_{i}.npy"), emb_list)
            emb_list = []
    
    # process last batch
    if len(sample_list)>0 or len(emb_list)>0:
        while len(sample_list) > 0:
            curr_batch.append(sample_list.pop())
        
        if len(curr_batch)>0:
            emb_list.append(extract_batch(curr_batch, meanpool=meanpool, mult_factor = mult_factor))
            
        emb_list = np.vstack(emb_list)
        i+=1
        np.save(os.path.join(out_dir, f"m{mult_factor}_{i}.npy"), emb_list)
        curr_batch = []
        emb_list = []
    
    return


def main(args):

    # Check if out directory exists, if not, create it
    os.makedirs(args.output_dir, exist_ok=True)

    # specify segmentation parameters     
    seg_dict = {
        "segment_length_s": args.seg_len,
        "cutoff": args.cutoff,
        "overlap": args.overlap,
        "target_sr": JUKEBOX_SR,
        "n_channels": 1,
        "normalize_loudness": True,
        "normalize_amplitude": True
    }
    
    # get file paths from sample dir
    file_paths = list_wavs_from_dir(args.samples_dir)

    extract_from_files(
        file_paths, args.output_dir, batch_size=args.batch_size, meanpool=args.meanpool,
        mult_factor=args.mult_factor, emb_chunk_size=args.chunk_size, **seg_dict
    )


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Parsing Arguments")

    # Add arguments
    parser.add_argument("--samples_dir", type=str, help="Path to directory containing audio samples.")
    parser.add_argument("--output_dir", type=str, help="Path to directory to save the embeddings to.")
    
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size, defaults to 4.")
    parser.add_argument("--meanpool", default=False, help="Wether to perform mean pooling.", action=argparse.BooleanOptionalAction)

    parser.add_argument("--mult_factor", default=100, help="Number of embeddings to be extracted from each audio segment. Defaults to 100. Meanpooling overrides this argument")
    parser.add_argument("--chunk_size", default=1000, help="Number of embedding chunks to save in each file for saving RAM. Defaults to 1000.")
    
    parser.add_argument("--seg_len", default=5, help="Duration of audio segments in seconds. Default is 5")
    parser.add_argument("--cutoff", type=str, default="crop",  help="Wether to ignore generated samples with length < seg_len, or pad them with silence. Can be ['crop', 'pad'].")
    parser.add_argument("--overlap", default=0.0, help="Percentage of overlap between samples. Default is 0.00.")

    # Parse the arguments
    args = parser.parse_args()

    main(args)
