from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import jukemirlib

from ..audio.audio_segmenter import segment_audio

JUKEBOX_SR = 44100
JKBX_WINDOW_LENGTH = 1048576
MAX_SEG_LEN_S = JKBX_WINDOW_LENGTH / JUKEBOX_SR


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
    assert mult_factor <= 1722, "Mult factor must be less than or equal to 1722!"

    embs = jukemirlib.extract(audio_samples, layers=[36], meanpool=meanpool)[36]

    # if batch_size == 1
    if embs.ndim == 2:
        embs = embs.reshape(1, embs.shape[0], embs.shape[1])

    if meanpool:
        final_embs = embs
    else:
        split_embeddings = np.array_split(embs, mult_factor, axis=1)
        mean_splits = [np.mean(arr, axis=1) for arr in split_embeddings]
        final_embs = np.vstack(mean_splits)

    return final_embs


def extract_segments(
    file_paths,
    out_dir,
    batch_size=4,
    meanpool=False,
    mult_factor=100,
    emb_chunk_size=1000,
    **segment_kwargs,
):
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

    print(
        f"Extracting embeddings to {out_dir} with a batch size of {batch_size} and m_factor of {mult_factor}"
    )

    i = 0
    emb_dict = {"embs": [], "meta": []}
    sample_dict = {"samples": [], "meta": []}
    batch_dict = {"samples": [], "meta": []}

    print(segment_kwargs)
    for f in tqdm(file_paths):
        curr_samples, curr_meta = segment_audio(f, **segment_kwargs)
        sample_dict["samples"].extend(curr_samples)
        sample_dict["meta"].extend(curr_meta)

        if len(sample_dict["samples"]) >= batch_size:
            while len(sample_dict["samples"]) >= batch_size:
                for _ in range(batch_size):
                    batch_dict["samples"].append(sample_dict["samples"].pop())
                    batch_dict["meta"].append(sample_dict["meta"].pop())

                emb_dict["embs"].append(
                    extract_batch(
                        batch_dict["samples"],
                        meanpool=meanpool,
                        mult_factor=mult_factor,
                    )
                )
                batch_dict["samples"] = []
                batch_dict["meta"] = []

        if len(emb_dict["embs"]) > emb_chunk_size:
            i += 1
            emb_dict["embs"] = np.vstack(emb_dict["embs"])
            np.save(os.path.join(out_dir, f"m{mult_factor}_{i}.npy"), emb_dict["embs"])
            emb_dict["embs"] = []

    # process last batch -> sample_list with < batch_size elements or emb_list with < emb_chunk_size elements
    if len(sample_dict["samples"]) > 0 or len(emb_dict["embs"]) > 0:
        while len(sample_dict["samples"]) > 0:
            batch_dict["samples"].append(sample_dict["samples"].pop())
            batch_dict["meta"].append(sample_dict["meta"].pop())

        if len(batch_dict["samples"]) > 0:
            n_batches = 0

            # curr_batch needs to be of shape (batch_size, seg_len, n_channels) because of precomputed TOP_PRIOR
            while len(batch_dict["samples"]) < batch_size:
                batch_dict["samples"].append(batch_dict["samples"][0])
                batch_dict["meta"].append(batch_dict["meta"][0])
                n_batches += 1

            emb_dict["embs"].append(
                extract_batch(
                    batch_dict["samples"], meanpool=meanpool, mult_factor=mult_factor
                )
            )

            # remove dummy batch from embs if it exists
            if n_batches > 0:
                emb_dict["embs"] = emb_dict["embs"][:-n_batches]
                emb_dict["meta"] = emb_dict["meta"][:-n_batches]

        i += 1
        emb_dict["embs"] = np.vstack(emb_dict["embs"])
        np.save(os.path.join(out_dir, f"m{mult_factor}_{i}.npy"), emb_dict["embs"])

    return


def extract_full_track(
    file_paths,
    out_dir,
    emb_chunk_size=100000,
    **segment_kwargs,
):
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

    print(f"Extracting one mean embedding per track to {out_dir}.")

    i = 0
    emb_dict = {"embs": [], "meta": []}

    print(segment_kwargs)

    for idx, f in enumerate(tqdm(file_paths)):
        # file_len_s = get_len_wavs([f])*60*60
        # segment_kwargs["segment_length_s"] = get_seg_len_fulltrack(file_len_s, MAX_SEG_LEN_S)

        # segment the audio
        curr_samples, curr_meta = segment_audio(f, **segment_kwargs)
        if len(curr_samples) == 0:
            print(f"No segments found for file {f}!")
            continue

        embs = []
        # extract each segment individually
        for samp in curr_samples:
            emb = extract_batch(samp, meanpool=True)
            embs.append(emb)

        # calculate mean of embeddings
        embs = np.mean(np.vstack(embs), axis=0)
        # add curr row idx to metadata
        curr_meta[0]["idx"] = idx
        curr_meta[0]["file_path"] = f

        # append data
        emb_dict["embs"].append(embs)
        emb_dict["meta"].append(curr_meta[0])

        if len(emb_dict["embs"]) > emb_chunk_size:
            i += 1
            emb_dict["embs"] = np.vstack(emb_dict["embs"])
            np.save(os.path.join(out_dir, f"emb_{i}.npy"), emb_dict["embs"])

            df = pd.DataFrame(emb_dict["meta"])
            df.to_csv(os.path.join(out_dir, f"meta_{i}.csv"))

            emb_dict["embs"] = []
            emb_dict["meta"] = []

    # save remaining embs
    i += 1

    emb_dict["embs"] = np.vstack(emb_dict["embs"])
    np.save(os.path.join(out_dir, f"emb_{i}.npy"), emb_dict["embs"])

    df = pd.DataFrame(emb_dict["meta"])
    df.to_csv(os.path.join(out_dir, f"meta_{i}.csv"))

    return
