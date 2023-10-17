from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import torch
from hear_mn import mn40_all_se_mel_avgs


from ..audio.audio_segmenter import segment_audio, get_seg_len_fulltrack
from ..audio.audio_utils import get_len_wavs

MN_SR = 32000
MN_MAX_SEG_LEN_S = 10


def extract_one_track(audio_samples, wrapper):
    print(len(audio_samples))
    audio_tensor = torch.stack([torch.from_numpy(audio) for audio in audio_samples])
    embs = mn40_all_se_mel_avgs.get_scene_embeddings(audio_tensor, wrapper)
    embs = embs.cpu().numpy()
    print(embs.shape)
    # embs = []
    # for audio in audio_samples:
    #     audio = torch.from_numpy(audio).unsqueeze(0)
    #     embed = mn40_all_se_mel_avgs.get_scene_embeddings(audio, wrapper)
    #     embed = embed.cpu().numpy()
    #     if np.isnan(embed).any():
    #         print("nan sample")
    #     embs.append(embed)

    # embs = np.mean(np.vstack(embs), axis=0)
    return embs


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
    wrapper = mn40_all_se_mel_avgs.load_model(model_name="mn40_as_ext").cuda()

    for idx, f in enumerate(tqdm(file_paths)):
        file_len_s = get_len_wavs([f]) * 60 * 60
        segment_kwargs["segment_length_s"] = get_seg_len_fulltrack(
            file_len_s, max_len=MN_MAX_SEG_LEN_S
        )

        # segment the audio
        curr_samples, curr_meta = segment_audio(f, **segment_kwargs)

        if len(curr_samples) == 0:
            print(f"No segments found for file {f}!")
            continue

        embs = extract_one_track(curr_samples, wrapper)

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
