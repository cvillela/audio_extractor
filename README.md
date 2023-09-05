# Audio Extractor
Repo configured with useful functions for preprocessing, segmenting and creating audio datasets, as well as extracting embeddings.

## Installation
 - Create a conda environment and activate it:
```
   conda create --name audioext
   conda activate audioext
```
- Be sure to have [torch>=2.0 and torchaudio>=2.0 installed](https://pytorch.org/get-started/locally/)
- Install the requirements
 ```
    pip install -r ./requirements.txt
 ```
- For jukebox embeddings - Install [Jukemirlib](https://github.com/rodrigo-castellon/jukemirlib)  from the github repo:
 ```
    pip install git+https://github.com/rodrigo-castellon/jukemirlib.git
 ```
- Install ffmpeg
 ```
   conda install -c conda-forge ffmpeg
 ```
## Current Features:
 - Efficiently segments audio files in given directory with length in seconds, arbitrary overlap between samples, padding/cropping segments, loudness and amplitude normalization.
 - Creates splits for train/test/val for segmented samples
 - Jukebox Embedding Extractor
   - (1 -1722) datapoints with (4800) dimensions for anything between (0-24) seconds of audio

## Jukebox Extractor
Extract [OpenAI's Jukebox](https://openai.com/research/jukebox) embeddings from a series of audio files contained in a directory.
From the project directory, run for help on the parameters: 
```
python -m audioext.pipelines.jukebox_extractor -h
```

### TO-DO
- [x] Reformat code to have a decoupled audio_segmenter utility
- [x] Reformat code to have a decoupled audio_processer utility
- [x] Jukebox extraction notebook to callable script
- [ ] Reformat pipelines for Dataset Generation.
- [ ] Reformat pipeline for Music Emotion Recognition.
- [ ] Create Audio Metadata enriching pipelines.   
- [ ] Add audio Denoise + Remove Silence.
- [ ] Add bandwith extension and audio super-resolution?

### Known bugs
 - Handle single audio file with varying sample_rate
 - Debug multi-channel normalization

