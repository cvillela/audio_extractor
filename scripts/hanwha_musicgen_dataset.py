
import os
import argparse
from pydub import AudioSegment
import splitfolders
from pathlib import Path
import shutil

from tqdm import tqdm
import uuid
import json

JSON_TEMPLATE = {
    "key": "",
    "artist": "hanwha",
    "sample_rate": None,
    "file_extension": "wav",
    "description": "hanwha, ",
    "keywords": "hanwha, ",
    "duration": 10.0,
    "bpm": "",
    "genre": "hanwha",
    "title": "hanwha",
    "name": "", #important!!!! matches wav file name
    "instrument": "hanwha",
    "moods": ['classic', 'gugak', 'hanwha']
}

def process_audio(file_path, file_name, dirname, output_dir_samples, output_dir_meta, segment_length_s=10, cutoff='pad', resample_32=True, mono=True, overlap=0.1):

    # Get file name without extension for caption
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    audio = AudioSegment.from_file(file_path)
    
    if resample_32:
        audio = audio.set_frame_rate(32000) # convert to 32kHz
    if mono:
        audio = audio.set_channels(1) # convert to mono
    
    sample_rate = audio.frame_rate
    segment_length_ms = segment_length_s * 1000
    step = int((1-overlap)*segment_length_ms)
    y = 0
    
    for i in range(0, len(audio), step):
        # create segment
        start_time = i 
        end_time = i + segment_length_ms
        segment = audio[start_time:end_time]

        # pad or crop end-of-file samples
        if end_time > len(audio):
            if cutoff=='pad':
                # pad at most 50% of the signal
                if len(segment) < 0.5*segment_length_ms:
                    break
                # Pad Length
                pad_len = segment_length_ms - len(segment)
                # Pad with 0s
                silence = AudioSegment.silent(duration=pad_len)
                segment = segment + silence
            elif cutoff=='leave':
                segment = audio[start_time:len(audio)]
            elif cutoff=='crop':
                break
            
        y+=1

        sample_name = str(uuid.uuid4())
        seg_path = os.path.join(output_dir_samples, sample_name+'.wav')
        
        # Save the segment
        segment.export(seg_path, format='wav')
        curr_json = JSON_TEMPLATE.copy()

        curr_json["sample_rate"] = sample_rate 
        curr_json["title"] = file_name.split('.')[0]
        curr_json["name"] = sample_name
        
        # Save the metadata
        meta_path = os.path.join(output_dir_meta, sample_name+'.json')
        
        with open(meta_path, 'w', encoding='utf8') as fp:
            json.dump(curr_json, fp)
    
    # print(f"Saved {y} segments from {file_path}")


def get_n_files(dir):
    for dirpath, dirnames, filenames in os.walk(dir):
        print(f"{dir} has {len(filenames)} files.")


def get_len_files(dir):
        dur=0
        for dirpath, dirnames, filenames in os.walk(dir):
            for file_name in tqdm(filenames):
                if file_name.endswith('.wav'):
                    audio = AudioSegment.from_file(os.path.join(dirpath,file_name))
                    dur+=len(audio)
        print(f"{dir} files have {dur/(1000*60*60)} hours.")    


def main(args):

    # Iterate through the files in the "samples" directory
    for dirpath, dirnames, filenames in os.walk(args.samples_dir):
        print(dirpath)
        for file_name in tqdm(filenames):
            if file_name.endswith('.wav'):
                process_audio(
                    file_path=os.path.join(dirpath, file_name),
                    file_name=file_name,
                    dirname=dirpath.split(os.path.sep)[-1],
                    output_dir_samples=args.output_dir_samples,
                    output_dir_meta=args.output_dir_metadata,
                    segment_length_s=args.seg_len,
                    cutoff=args.cutoff,
                    resample_32=args.resample_32,
                    mono=args.mono,
                    overlap=args.overlap,
                )
    # print statistics
    get_len_files(args.samples_dir)
    get_n_files(args.samples_dir)
    print()
    get_n_files(args.output_dir_samples)
    get_len_files(args.output_dir_samples)
    print()
    get_n_files(args.output_dir_metadata)

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Parsinnnn")

    # Add arguments
    parser.add_argument("--samples_dir", type=str, default='/home/cvillela/dataland/data/hanwha/validation/source', help="Path to directory containing raw samples. Defaults to ~/dataland/data/hanwha/training/source.")
    parser.add_argument("--output_dir_samples", type=str, default='/home/cvillela/dataland/data/hanwha/validation/ds/samples', help="Path to directory to save the cropped samples. Defaults to /home/cvillela/dataland/data/training/ds.")
    parser.add_argument("--output_dir_metadata", type=str, default='/home/cvillela/dataland/data/hanwha/validation/ds/metadata', help="Path to directory to save the corresponding metadata. Defaults to /home/cvillela/dataland/data/training/ds/metadata")

    parser.add_argument("--seg_len", default=10, help="Duration of audio segments in seconds. Default is 10")
    
    parser.add_argument("--cutoff",       type=str, default="leave",  help="Wether to ignore generated samples with length < seg_len, or pad them with silence. Default is 'crop', can also be 'pad' or 'leave'.")
    parser.add_argument("--resample_32",  default=True, help="Wether to resample the segments to 32000Hz (for MusicGen training). Default True")
    parser.add_argument("--mono",         default=True, help="Wether to transform audio from stereo to mono. Default True")
    parser.add_argument("--overlap",      default=0.15, help="Percentage of overlap between samples. Default is 0.15.")

    # Parse the arguments
    args = parser.parse_args()

    # Check if directories exists, if not, create 'em
    if not os.path.exists(args.output_dir_samples):
        print('Creating output sample dir')
        os.makedirs(args.output_dir_samples)
    if not os.path.exists(args.output_dir_metadata):
        print('Creating output metadata dir')
        os.makedirs(args.output_dir_metadata)
    
    main(args)

