import os
from pydub import AudioSegment, silence, effects
import uuid
from unidecode import unidecode
import json
import numpy as np
import scipy.io.wavfile
import splitfolders
from tqdm import tqdm
import shutil
import math

from .audio_processer import (
    normalize_unit,
    normalize_loudness,
    resample_segment,
    set_channels,
)
from .audio_processer import audiosegment_to_ndarray_32


def get_audio_metadata(audio, filename):
    """
    Generate the audio metadata for the given audio.

    Parameters:
        audio (AudioSegment): The audio segment for which to retrieve the metadata.

    Returns:
        dict: A dictionary containing the audio metadata with the following keys:
            - sample_rate (int): The sample rate of the audio.
            - channels (int): The number of audio channels.
            - bytes_per_sample (int): The number of bytes per audio sample. A value of 1 indicates 8 bit, and a value of 2 indicates 16 bit.
    """
    meta = {
        "sample_rate": audio.frame_rate,
        "channels": audio.channels,
        "bytes_per_sample": audio.sample_width,  # 1 means 8 bit, 2 means 16 bit
        "filename": filename,
        "duration": audio.duration_seconds,
    }

    return meta


def segment_audio(
    file_path,
    segment_length_s=10,
    target_sr=32000,
    n_channels=1,
    cutoff="pad",
    overlap=0.0,
    loudness_norm=True,
    normalize_amplitude=True,
):
    """
    Segment an audio file into smaller segments.

    Args:
        file_path (str): The path to the audio file.
        segment_length_s (float, optional): The duration of each segment in seconds. Defaults to 10.
        target_sr (int, optional): The target sample rate for resampling. Defaults to 32000.
        n_channels (int, optional): The number of audio channels. Defaults to 1.
        cutoff (str, optional): The strategy for handling segments that are shorter than segment_length_s.
                               Possible values are 'pad', 'leave', and 'crop'. Defaults to 'pad'.
        overlap (float, optional): The amount of overlap between segments as a fraction of segment_length_s.
                                   Defaults to 0.0.
        normalize_loudness (bool, optional): Whether to normalize the loudness of the audio. Defaults to True.
        normalize_amplitude (bool, optional): Whether to normalize the amplitude of the audio. Defaults to True.

    Returns:
        list: A list of segmented audio as np.float32 and a list of corresponding metadata.
    """

    # Get file name without extension for caption
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    file_name = unidecode(file_name.lower())

    # load audio
    audio = AudioSegment.from_file(file_path)

    # resample
    if target_sr is not None:
        audio = resample_segment(audio, target_sr)  # resample to target_sr

    # force mono/stereo
    if n_channels is not None:
        audio = set_channels(audio, n_channels)  # convert to mono

    # normalize loudness
    if loudness_norm:
        audio = normalize_loudness(audio)

    # Audio info dict
    audio_metadata = get_audio_metadata(audio, file_name)

    # turn into np array
    np_audio = audiosegment_to_ndarray_32(audio)

    if normalize_amplitude:
        np_audio = normalize_unit(np_audio)

    segment_list = []
    meta_list = []

    # segment into segment_length_s samples
    segment_length_samples = math.ceil(segment_length_s * audio.frame_rate)
    step = int((1 - overlap) * segment_length_samples)
    for i in range(0, len(np_audio), step):
        # create segment
        start_time = i
        end_time = i + segment_length_samples
        segment = np_audio[start_time:end_time]

        # pad or crop end-of-file samples
        if end_time > len(np_audio):
            if cutoff == "pad":  # pad with silence (0 amplitude)
                if (
                    len(segment) < 0.8 * segment_length_samples
                ):  # pad at most 20% of the signal
                    continue

                # Pad with 0s
                pad_len = segment_length_samples - len(segment)
                pad = np.zeros(pad_len)
                segment = np.concatenate((segment, pad), axis=0)

            elif cutoff == "leave":  # save sample smaller than segment_length_s
                segment = np_audio[start_time : len(np_audio)]

            elif cutoff == "crop":  # discard smaller sample
                continue

        segment = segment.astype(np.float32)

        segment_list.append(segment)
        meta_list.append(audio_metadata)

    return segment_list, meta_list


def save_sample_meta(audio_meta, segment, out_dir):
    """
    Save the audio segment and its metadata to the specified output directory.

    Args:
        audio_meta (dict): Metadata for the audio segment.
        segment (AudioSegment): Audio segment to be saved.
        out_dir (str): Output directory where the segment and metadata will be saved.
    """
    # declare metadata and sample dirs
    sample_dir = os.path.join(out_dir, "samples")
    meta_dir = os.path.join(out_dir, "metadata")

    # create sample and metadata directories if they don't exist
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    sample_name = str(uuid.uuid4())
    seg_path = os.path.join(sample_dir, f"{sample_name}.wav")
    meta_path = os.path.join(meta_dir, f"{sample_name}.json")

    audio_meta["name"] = sample_name

    # Save the segment as a WAV file
    scipy.io.wavfile.write(seg_path, audio_meta["sample_rate"], segment)

    # Save the metadata as a JSON file
    with open(meta_path, "w", encoding="utf8") as fp:
        json.dump(audio_meta, fp)


def generate_splits(out_dir, val_ratio=0.2, n_test=50, no_classes=True):
    # here goes some REALLY UGLY code, if you are reading, I am sorry.

    sample_dir = os.path.join(out_dir, "samples")
    meta_dir = os.path.join(out_dir, "metadata")

    if no_classes:
        print("Moving all files in sample dir to dummy folder...")
        os.makedirs(os.path.join(sample_dir, "dummy"), exist_ok=True)
        # move all sample_dir files to a dummy folder inside sample_dir
        for f in tqdm(os.listdir(sample_dir)):
            if f != "dummy":
                shutil.move(
                    os.path.join(sample_dir, f), os.path.join(sample_dir, "dummy", f)
                )

    splits_dir = os.path.join(out_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    print("Creating random train/val splits!")
    splitfolders.ratio(
        sample_dir, output=splits_dir, seed=1337, ratio=(1 - val_ratio, val_ratio)
    )

    # create test directory
    os.makedirs(os.path.join(splits_dir, "test"), exist_ok=True)

    print(f"Moving {n_test} files from train to test...")
    # move n_test random files in train dir to test dir
    for f in tqdm(os.listdir(os.path.join(splits_dir, "train", "dummy"))[:n_test]):
        shutil.move(
            os.path.join(splits_dir, "train", "dummy", f),
            os.path.join(splits_dir, "test"),
        )

    print("Copying metadata to corresponding splits...")
    # move metadata to split folders
    for dirpath, dirnames, filenames in os.walk(splits_dir):
        for file_name in tqdm(filenames):
            fn = file_name.split(".")[0] + ".json"
            shutil.copy(os.path.join(meta_dir, fn), os.path.join(dirpath, fn))

    if no_classes:
        print("Deleting dummy folders in sample directory...")
        # move all files in sample_dir/dummy back to sample_dir and delete dummy dir
        for f in tqdm(os.listdir(os.path.join(sample_dir, "dummy"))):
            if f != "dummy":
                shutil.move(
                    os.path.join(sample_dir, "dummy", f), os.path.join(sample_dir)
                )
        os.rmdir(os.path.join(sample_dir, "dummy"))

        # repeat for training, validation in splits directory
        dir_list = ["train", "val"]
        for d in dir_list:
            for f in tqdm(os.listdir(os.path.join(splits_dir, d, "dummy"))):
                if f != "dummy":
                    shutil.move(
                        os.path.join(splits_dir, d, "dummy", f),
                        os.path.join(splits_dir, d),
                    )
            os.rmdir(os.path.join(splits_dir, d, "dummy"))

    print("Splits created!")

    return


def get_seg_len_fulltrack(audio_len, max_len=25):
    """
    Calculate the length of a segment for full track extraction, based on the given audio length and maximum segment length.

    Parameters:
    - audio_len (int): The total length of the audio in seconds.
    - max_len (int): The maximum length of a segment in seconds.

    Returns:
    - float: The length of a segment based on the given audio length and maximum length in seconds.
    """
    if audio_len < max_len:
        # return math.ceil(audio_len)
        return audio_len
    else:
        i = 2
        while audio_len / i > max_len:
            i += 1
        # return math.ceil(audio_len/i)
        return audio_len / i
